#!/usr/bin/env python3
"""
Loss-based head attribution for Dream (layer-wise Integrated Gradients over per-head gates).

This mirrors the LLaDA loss-attribution implementation, but is adapted to Dream's HF-style layout:
  - transformer layers: model.model.layers (ModuleList[DreamDecoderLayer])
  - attention output projection: layer.self_attn.o_proj

Objective (diffusion-aligned masked LM loss):
  - Build a full sequence = prompt + completion
  - Randomly mask a *portion* of completion tokens (multiple mask_probs, Monte-Carlo samples)
  - Compute CE loss only on masked completion positions; average over mask variants
  - Compute IG per-layer over a per-head gate α inserted right before this layer's o_proj

Outputs a head_importance.pt compatible with the evaluation pipeline:
  {
    "importance_scores": {layer_idx: tensor[n_heads]},
    "metadata": {...}
  }
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.checkpoint import checkpoint

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Ensure repo root is on sys.path (so `import models.*` works when running directly)
# File path: adaptive-dllm/models/Dream/attribution/loss_attribution/compute_loss_attribution.py
# Go up 4 levels => adaptive-dllm/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.Dream.core.modeling_dream import DreamModel


def _stable_int_seed(seed: int, *parts: int) -> int:
    """
    Robust deterministic seed mixer (avoids collisions better than linear combos).
    Returns a 31-bit non-negative int (works for torch.Generator.manual_seed).
    """
    h = hashlib.sha1()
    h.update(str(int(seed)).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(int(p)).encode("utf-8"))
    return int(h.hexdigest()[:8], 16) & 0x7FFFFFFF


def _find_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Dream layout:
      - DreamModel.model is DreamBaseModel
      - DreamBaseModel.layers is a ModuleList of decoder layers
    """
    m = model
    if hasattr(m, "model") and isinstance(getattr(m, "model"), torch.nn.Module):
        base = getattr(m, "model")
        if hasattr(base, "layers") and isinstance(getattr(base, "layers"), torch.nn.ModuleList):
            return list(getattr(base, "layers"))
    if hasattr(m, "layers") and isinstance(getattr(m, "layers"), torch.nn.ModuleList):
        return list(getattr(m, "layers"))
    raise AttributeError("Could not locate Dream transformer layers (expected model.model.layers).")


def _find_attn_and_oproj(layer: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module]:
    if not hasattr(layer, "self_attn"):
        raise AttributeError("Dream layer has no `self_attn`.")
    attn = getattr(layer, "self_attn")
    if not hasattr(attn, "o_proj"):
        raise AttributeError("Dream attention has no `o_proj`.")
    o_proj = getattr(attn, "o_proj")
    if not isinstance(o_proj, torch.nn.Module):
        raise TypeError("Dream attention `o_proj` is not a torch.nn.Module.")
    return attn, o_proj


def _get_num_heads(attn: torch.nn.Module, fallback_from_config: int | None = None) -> int:
    for name in ("num_heads", "n_heads", "n_head", "heads"):
        if hasattr(attn, name):
            v = getattr(attn, name)
            if isinstance(v, int) and v > 0:
                return int(v)
    if hasattr(attn, "config") and hasattr(attn.config, "num_attention_heads"):
        v = getattr(attn.config, "num_attention_heads")
        if isinstance(v, int) and v > 0:
            return int(v)
    if fallback_from_config is not None:
        return int(fallback_from_config)
    raise AttributeError("Could not infer num_heads from Dream attention module.")


def _parse_gsm8k_final_answer(answer_str: str) -> str:
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    return answer_str.strip()


def _build_gsm8k_prompt_and_completion(
    question: str,
    answer: str,
    *,
    tokenizer,
    use_chat_template: bool,
    completion_mode: str,
) -> Tuple[str, str]:
    """
    Build GSM8K prompt + completion for attribution.
    
    IMPORTANT:
    - Dream eval uses chat formatting (apply_chat_template). If attribution uses a plain prompt while
      eval uses chat, the distribution mismatch can make importance scores ineffective.
    - completion_mode:
        - "final": supervise only the final numeric answer (shorter; default)
        - "full":  supervise the full GSM8K reference answer (often closer to model outputs)
    """
    base_user = f"Question: {question}\nAnswer:"
    if completion_mode == "final":
        final = _parse_gsm8k_final_answer(answer)
        completion = f" {final}"
    elif completion_mode == "full":
        ans = answer if isinstance(answer, str) else str(answer)
        completion = ans if ans.startswith((" ", "\n")) else f" {ans}"
    else:
        raise ValueError(f"Unsupported completion_mode: {completion_mode}")

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_user},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt, completion
        except Exception:
            pass

    return base_user, completion


def _prepare_nemotron_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    system = sample.get("system_prompt", None)
    if isinstance(system, str) and system.strip():
        msgs.append({"role": "system", "content": system.strip()})

    input_data = sample.get("input", None)
    if isinstance(input_data, list):
        for m in input_data:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user")).strip() or "user"
            content = str(m.get("content", "")).strip()
            if content:
                msgs.append({"role": role, "content": content})
    elif isinstance(input_data, str) and input_data.strip():
        msgs.append({"role": "user", "content": input_data.strip()})

    if not msgs:
        msgs = [{"role": "user", "content": "Hello"}]
    return msgs


def _prepare_nemotron_prompt(sample: Dict[str, Any]) -> str:
    prompt_parts: List[str] = []
    system = sample.get("system_prompt", None)
    if isinstance(system, str) and system.strip():
        prompt_parts.append(f"System: {system.strip()}")

    input_data = sample.get("input", None)
    if isinstance(input_data, list):
        for msg in input_data:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            if role and content:
                prompt_parts.append(f"{role}: {content}")
    elif isinstance(input_data, str) and input_data.strip():
        prompt_parts.append(input_data.strip())

    prompt = "\n\n".join(prompt_parts).strip()
    return prompt if prompt else "Hello"


def _build_nemotron_prompt_and_completion(
    sample: Dict[str, Any],
    *,
    tokenizer,
    use_chat_template: bool,
) -> Tuple[str, str]:
    output = sample.get("output", "")
    if not isinstance(output, str):
        output = str(output)
    completion = output if output.startswith((" ", "\n")) else f" {output}"

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = _prepare_nemotron_messages(sample)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt, completion
        except Exception:
            # Fall back to plain prompt
            pass

    prompt = _prepare_nemotron_prompt(sample)
    return prompt, completion


def _tokenize_pair(
    tokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Returns:
      input_ids: (1, L)
      attention_mask: (1, L)
      completion_start: int (index where completion span starts)
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(completion, add_special_tokens=False).input_ids

    # If prompt already contains chat special tokens (e.g., "<|im_start|>system"),
    # DO NOT prepend BOS. This keeps attribution tokenization aligned with eval prompts.
    bos = []
    if (tokenizer.bos_token_id is not None) and ("<|im_start|>" not in prompt):
        bos = [tokenizer.bos_token_id]

    input_ids_list = (bos + prompt_ids + answer_ids)[:max_length]
    completion_start = min(len(bos) + len(prompt_ids), len(input_ids_list))

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    # IMPORTANT (Dream):
    # Dream's SDPA attention expects `attention_mask` to be either:
    # - None/"full" (no mask), or
    # - a properly shaped bool/float attn_mask (e.g., [B, 1, T, T]).
    # Passing a standard HF-style (B, T) int/long mask will be forwarded into SDPA as-is and crash,
    # especially under bf16 autocast. Since we do not pad here, we can safely omit attention_mask.
    attention_mask = None
    return input_ids, attention_mask, int(completion_start)


def _masked_ce_answer_only_batch(logits: torch.Tensor, labels: torch.Tensor, *, normalize: str) -> torch.Tensor:
    """
    logits: (B, T, V)
    labels: (B, T) with -100 ignored
    """
    # IMPORTANT:
    # We compute loss per-sample, then average across batch.
    # This avoids subtle scaling bugs when we later chunk `n_variants` and aggregate across chunks.
    b, t, v = logits.shape
    token_losses = F.cross_entropy(
        logits.view(-1, v),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(b, t)
    mask = labels != -100
    masked_losses = token_losses * mask.to(token_losses.dtype)
    per_sample_sum = masked_losses.sum(dim=1)  # (B,)
    if normalize == "sum":
        return per_sample_sum.mean()
    if normalize == "mean_masked":
        denom = mask.sum(dim=1).clamp(min=1).to(per_sample_sum.dtype)  # (B,)
        per_sample_mean = per_sample_sum / denom
        return per_sample_mean.mean()
    raise ValueError(f"Unsupported normalize mode: {normalize}")


def _build_labels_and_masked_inputs_for_completion_span(
    *,
    full_input_ids: torch.Tensor,  # (1, L)
    completion_start: int,
    mask_token_id: int,
    mask_prob: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask a portion of completion tokens.
    Labels supervise ONLY the masked completion tokens; others are -100.
    """
    ids = full_input_ids[0]  # (L,)
    L = int(ids.numel())
    start = int(completion_start)
    if start >= L:
        labels_1d = torch.full((L,), -100, dtype=torch.long, device=ids.device)
        return ids.unsqueeze(0), labels_1d.unsqueeze(0)

    span_len = L - start
    if span_len <= 0:
        labels_1d = torch.full((L,), -100, dtype=torch.long, device=ids.device)
        return ids.unsqueeze(0), labels_1d.unsqueeze(0)

    # Sample mask positions inside completion span
    probs = torch.full((span_len,), float(mask_prob), device=ids.device, dtype=torch.float32)
    u = torch.rand((span_len,), generator=generator, device=ids.device, dtype=torch.float32)
    mask = u < probs  # (span_len,)
    
    # Ensure at least one token is supervised (avoid zero-loss variants)
    if mask_prob > 0.0 and not bool(mask.any()):
        j = int(torch.randint(low=0, high=span_len, size=(1,), device=ids.device, generator=generator).item())
        mask[j] = True

    labels_1d = torch.full((L,), -100, dtype=torch.long, device=ids.device)
    if bool(mask.any()):
        idx = torch.nonzero(mask, as_tuple=False).view(-1) + start
        labels_1d[idx] = ids[idx]
    input_ids_masked_1d = ids.clone()
    if bool(mask.any()):
        input_ids_masked_1d[idx] = int(mask_token_id)
    return input_ids_masked_1d.unsqueeze(0), labels_1d.unsqueeze(0)


def _get_mask_token_id(model, tokenizer) -> int:
    tok_id = getattr(tokenizer, "mask_token_id", None)
    if isinstance(tok_id, int) and tok_id >= 0:
        return int(tok_id)
    cfg_id = getattr(getattr(model, "config", None), "mask_token_id", None)
    if isinstance(cfg_id, int) and cfg_id >= 0:
        return int(cfg_id)
    # Dream default in configuration_dream.py
    return 151666


@torch.no_grad()
def _dry_run_check_o_proj_shape(o_proj: torch.nn.Module, hidden_size: int) -> None:
    if hasattr(o_proj, "in_features") and isinstance(o_proj.in_features, int):
        if int(o_proj.in_features) != int(hidden_size):
            raise ValueError(f"o_proj.in_features={o_proj.in_features} != hidden_size={hidden_size}")


def compute_layer_ig(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    layer_idx: int,
    tokenizer,
    dataset_rows: List[Dict[str, Any]],
    *,
    device: torch.device,
    ig_steps: int,
    baseline_value: float,
    max_length: int,
    num_heads_from_config: int,
    use_amp_bf16: bool,
    dataset_name: str,
    dataset_use_chat_template: bool,
    gsm8k_completion_mode: str,
    mask_probs: List[float],
    mask_samples_per_prob: int,
    loss_normalize: str,
    seed: int,
    ig_postprocess: str,
    mask_batch_size: int,
    show_progress: bool,
    progress_update_every: int,
    debug_gate: bool = False,
    gradient_checkpointing: bool = False,
) -> torch.Tensor:
    attn, o_proj = _find_attn_and_oproj(layer)
    n_heads = _get_num_heads(attn, fallback_from_config=num_heads_from_config)

    if not hasattr(o_proj, "in_features"):
        raise AttributeError("o_proj has no in_features; cannot infer head_dim safely.")
    hidden_size = int(o_proj.in_features)
    if hidden_size % n_heads != 0:
        raise ValueError(f"hidden_size={hidden_size} not divisible by n_heads={n_heads}")
    head_dim = hidden_size // n_heads
    _dry_run_check_o_proj_shape(o_proj, hidden_size)

    class _OProjHeadGate:
        def __init__(self, o_proj_module: torch.nn.Module, *, n_heads: int, head_dim: int):
            self.o_proj = o_proj_module
            self.n_heads = int(n_heads)
            self.head_dim = int(head_dim)
            self.alpha: Optional[torch.Tensor] = None
            self._handle = None

        def _pre_hook(self, module, inputs):
            if module is not self.o_proj:
                return inputs
            alpha = self.alpha
            if alpha is None:
                return inputs
            x = inputs[0]  # (B, T, hidden_size)
            b, t, hs = x.shape
            x_ = x.view(b, t, self.n_heads, self.head_dim)
            x_ = x_ * alpha.view(1, 1, self.n_heads, 1)
            x_ = x_.view(b, t, hs)
            return (x_,) + tuple(inputs[1:])

        def install(self) -> None:
            if self._handle is not None:
                raise RuntimeError("Gate hook already installed.")
            self._handle = self.o_proj.register_forward_pre_hook(self._pre_hook)

        def remove(self) -> None:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None

    gate = _OProjHeadGate(o_proj, n_heads=n_heads, head_dim=head_dim)
    gate.install()
    try:
        def _forward_logits(input_ids: torch.Tensor) -> torch.Tensor:
            """
            Forward helper.
            NOTE: We always use the input_ids path here (fast). Gradient checkpointing, if enabled,
            is handled inside the Dream model (we configure it in main()).
            """
            return model(input_ids, num_logits_to_keep=num_logits_to_keep).logits

        ig_sum = torch.zeros(n_heads, device=device, dtype=torch.float32)
        total_items = 0

        scale = float(1.0 - baseline_value)
        did_debug_check = False
        mask_token_id = _get_mask_token_id(model, tokenizer)

        iterator = enumerate(dataset_rows)
        # In nohup/non-tty logs, tqdm can appear "stuck" (carriage returns), so we only use it on a real tty.
        use_tqdm = bool(show_progress and (tqdm is not None) and sys.stderr.isatty())
        if use_tqdm:
            iterator = tqdm(  # type: ignore[assignment]
                iterator,
                total=len(dataset_rows),
                desc=f"layer {layer_idx}",
                dynamic_ncols=True,
                leave=False,
            )

        for row_idx, row in iterator:
            if dataset_name == "gsm8k":
                prompt, completion = _build_gsm8k_prompt_and_completion(
                    row["question"],
                    row["answer"],
                    tokenizer=tokenizer,
                    use_chat_template=dataset_use_chat_template,
                    completion_mode=gsm8k_completion_mode,
                )
            elif dataset_name == "nemotron":
                prompt, completion = _build_nemotron_prompt_and_completion(
                    row, tokenizer=tokenizer, use_chat_template=dataset_use_chat_template
                )
            else:
                raise ValueError(f"Unsupported dataset_name: {dataset_name}")

            full_input_ids, attention_mask, completion_start = _tokenize_pair(
                tokenizer,
                prompt,
                completion,
                device=device,
                max_length=max_length,
            )

            # DreamModel.forward supports `num_logits_to_keep`, which computes logits only for the
            # last K tokens. Since we supervise only completion tokens (which are at the end),
            # we can safely compute logits only for the completion span to save memory.
            num_logits_to_keep = int(full_input_ids.size(1) - int(completion_start))
            if num_logits_to_keep <= 0:
                continue

            masked_batches: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]] = []
            for prob_idx, prob in enumerate(mask_probs):
                for s in range(int(mask_samples_per_prob)):
                    gen = torch.Generator(device=device)
                    gen.manual_seed(_stable_int_seed(int(seed), int(layer_idx), int(row_idx), int(prob_idx), int(s)))
                    input_ids_masked, labels = _build_labels_and_masked_inputs_for_completion_span(
                        full_input_ids=full_input_ids,
                        completion_start=completion_start,
                        mask_token_id=mask_token_id,
                        mask_prob=float(prob),
                        generator=gen,
                    )
                    if (labels != -100).sum().item() <= 0:
                        continue
                    masked_batches.append((input_ids_masked, attention_mask, labels))

            if len(masked_batches) == 0:
                continue

            if debug_gate and (not did_debug_check):
                with torch.no_grad():
                    gate.alpha = torch.ones(n_heads, device=device, dtype=torch.float32)
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            if masked_batches[0][1] is None:
                                logits1 = _forward_logits(masked_batches[0][0])
                            else:
                                logits1 = model(
                                    masked_batches[0][0],
                                    attention_mask=masked_batches[0][1],
                                    num_logits_to_keep=num_logits_to_keep,
                                ).logits
                    else:
                        if masked_batches[0][1] is None:
                            logits1 = _forward_logits(masked_batches[0][0])
                        else:
                            logits1 = model(
                                masked_batches[0][0],
                                attention_mask=masked_batches[0][1],
                                num_logits_to_keep=num_logits_to_keep,
                            ).logits

                    alpha2 = torch.ones(n_heads, device=device, dtype=torch.float32)
                    alpha2[0] = 0.0
                    gate.alpha = alpha2
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            if masked_batches[0][1] is None:
                                logits2 = _forward_logits(masked_batches[0][0])
                            else:
                                logits2 = model(
                                    masked_batches[0][0],
                                    attention_mask=masked_batches[0][1],
                                    num_logits_to_keep=num_logits_to_keep,
                                ).logits
                    else:
                        if masked_batches[0][1] is None:
                            logits2 = _forward_logits(masked_batches[0][0])
                        else:
                            logits2 = model(
                                masked_batches[0][0],
                                attention_mask=masked_batches[0][1],
                                num_logits_to_keep=num_logits_to_keep,
                            ).logits

                    delta = (logits1.to(torch.float32) - logits2.to(torch.float32)).abs().mean().item()
                    if delta <= 0.0:
                        raise RuntimeError(
                            "Debug gate check failed: changing alpha did not change logits. "
                            "Hook may not be attached to the active o_proj module."
                        )
                    print(f"[debug_gate] layer={layer_idx} mean|Δlogits|={delta:.6g}")
                    did_debug_check = True

            grads_accum = torch.zeros(n_heads, device=device, dtype=torch.float32)

            all_input_ids = torch.cat([x for (x, _, _) in masked_batches], dim=0)
            # In Dream loss attribution we do not pass attention_mask (see _tokenize_pair note).
            all_labels = torch.cat([y for (_, _, y) in masked_batches], dim=0)
            n_variants = int(all_input_ids.size(0))
            # Align labels with logits when we use `num_logits_to_keep` (DreamModel returns last-K logits).
            all_labels_tail = all_labels[:, -num_logits_to_keep:]

            chunk = int(mask_batch_size)
            if chunk <= 0:
                chunk = n_variants

            for k in range(1, ig_steps + 1):
                t = float(k) / float(ig_steps)
                alpha_val = baseline_value + t * (1.0 - baseline_value)
                alpha = torch.full(
                    (n_heads,),
                    fill_value=float(alpha_val),
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                gate.alpha = alpha

                model.zero_grad(set_to_none=True)

                # Average loss over ALL masked variants (avoid chunk-size bias)
                loss_weighted_sum: Optional[torch.Tensor] = None
                total_variants = 0
                if use_amp_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        for start in range(0, n_variants, chunk):
                            end = min(start + chunk, n_variants)
                            logits = _forward_logits(all_input_ids[start:end])
                            l = _masked_ce_answer_only_batch(
                                logits, all_labels_tail[start:end], normalize=loss_normalize
                            )
                            bsz = int(end - start)
                            loss_weighted_sum = l * bsz if loss_weighted_sum is None else (loss_weighted_sum + l * bsz)
                            total_variants += bsz
                else:
                    for start in range(0, n_variants, chunk):
                        end = min(start + chunk, n_variants)
                        logits = _forward_logits(all_input_ids[start:end])
                        l = _masked_ce_answer_only_batch(
                            logits, all_labels_tail[start:end], normalize=loss_normalize
                        )
                        bsz = int(end - start)
                        loss_weighted_sum = l * bsz if loss_weighted_sum is None else (loss_weighted_sum + l * bsz)
                        total_variants += bsz

                if loss_weighted_sum is None or total_variants <= 0:
                    continue

                loss = loss_weighted_sum / float(total_variants)
                loss.backward()

                if alpha.grad is None:
                    # Rare but possible (e.g., constant loss). Skip this step.
                    continue

                grads_accum += alpha.grad.detach().to(torch.float32)
                
                # Explicitly free memory after each IG step
                del loss, loss_weighted_sum
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            ig_row = scale * (grads_accum / float(max(ig_steps, 1)))
            if ig_postprocess == "abs":
                ig_row = ig_row.abs()
            elif ig_postprocess == "relu":
                ig_row = torch.relu(ig_row)
            elif ig_postprocess == "signed":
                pass
            else:
                raise ValueError(f"Unsupported ig_postprocess: {ig_postprocess}")

            ig_sum += ig_row
            total_items += 1
            
            # Free memory after processing this row
            del all_input_ids, all_labels, all_labels_tail, grads_accum, ig_row
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # If user asked for progress but tqdm isn't available, fall back to periodic prints.
            if show_progress and (not use_tqdm) and (progress_update_every > 0) and ((row_idx + 1) % int(progress_update_every) == 0):
                print(f"[layer {layer_idx}] processed {row_idx+1}/{len(dataset_rows)} rows ...")

        if total_items <= 0:
            return torch.zeros(n_heads, device=device, dtype=torch.float32)
        return (ig_sum / float(total_items)).detach().to(torch.float32)
    finally:
        gate.remove()


def _load_gsm8k_rows(*, split: str, max_samples: int, seed: int, dataset_shuffle: bool) -> List[Dict[str, Any]]:
    ds = load_dataset("gsm8k", "main", split=split)
    if dataset_shuffle:
        ds = ds.shuffle(seed=int(seed))
    rows = []
    for i, r in enumerate(ds):
        rows.append({"question": r["question"], "answer": r["answer"]})
        if len(rows) >= int(max_samples):
            break
    return rows


def _load_nemotron_rows(
    *,
    categories: List[str],
    samples_per_category: int,
    pool_per_category: int,
    data_seed: int,
    max_samples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    g = torch.Generator(device="cpu")
    g.manual_seed(int(data_seed))

    for cat in categories:
        ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=cat, streaming=True)
        pool: List[Dict[str, Any]] = []
        for i, sample in enumerate(ds):
            pool.append(sample)
            if len(pool) >= int(pool_per_category):
                break
        if len(pool) == 0:
            continue

        perm = torch.randperm(len(pool), generator=g).tolist()
        take = min(int(samples_per_category), len(pool))
        picked = [pool[idx] for idx in perm[:take]]
        rows.extend(picked)

    # cap
    if int(max_samples) > 0:
        rows = rows[: int(max_samples)]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Dream model path or HF id.")
    parser.add_argument("--output_dir", required=True, help="Directory to write head_importance.pt")

    parser.add_argument("--dataset", default="nemotron", choices=["nemotron", "gsm8k"])
    parser.add_argument("--split", default="test", help="gsm8k split (default: test)")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--dataset_shuffle", action="store_true", help="Shuffle dataset before taking max_samples.")

    # Nemotron controls
    parser.add_argument("--samples_per_category", type=int, default=10)
    parser.add_argument("--nemotron_pool_per_category", type=int, default=1000)
    parser.add_argument("--nemotron_categories", type=str, default="code,math,science,chat,safety")
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Build prompt using tokenizer.apply_chat_template(add_generation_prompt=True) when available (recommended for Dream-Instruct).",
    )
    parser.add_argument(
        "--gsm8k_completion_mode",
        type=str,
        default="final",
        choices=["final", "full"],
        help="For gsm8k: completion supervision mode. 'final' uses only final numeric answer; 'full' uses full reference answer.",
    )

    # IG controls
    parser.add_argument("--ig_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--baseline", type=str, default="zero", choices=["zero", "scalar"])
    parser.add_argument("--baseline_scalar", type=float, default=0.3)

    # Masking / diffusion-style multi-timestep
    parser.add_argument("--mask_probs", type=str, default="0.15,0.3,0.5,0.7,0.9")
    parser.add_argument("--mask_samples_per_prob", type=int, default=2)
    parser.add_argument("--loss_normalize", type=str, default="mean_masked", choices=["mean_masked", "sum"])
    parser.add_argument("--ig_postprocess", type=str, default="signed", choices=["abs", "signed", "relu"])
    parser.add_argument("--mask_batch_size", type=int, default=1)

    # Layer range
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=-1, help="-1 means last layer")

    # Seeds
    parser.add_argument("--seed", type=int, default=47, help="Overall seed (fallback).")
    parser.add_argument("--data_seed", type=int, default=None, help="Controls dataset subsampling/shuffle.")
    parser.add_argument("--mask_seed", type=int, default=None, help="Controls random masking positions.")

    # Perf
    parser.add_argument("--use_amp_bf16", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--progress_update_every", type=int, default=20)
    parser.add_argument("--debug_gate", action="store_true")

    # Checkpointing (Dream's built-in gradient_checkpointing only triggers under training mode)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable Dream gradient checkpointing (will switch model to train() during attribution to activate).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seed = int(args.seed)
    data_seed = int(args.data_seed) if args.data_seed is not None else seed
    mask_seed = int(args.mask_seed) if args.mask_seed is not None else seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========================================================")
    print("Dream Loss Attribution (Layer-wise Gate IG)")
    print("========================================================")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"device={device}")
    print(f"model_path={args.model_path}")
    print(f"dataset={args.dataset} max_samples={args.max_samples} max_length={args.max_length}")
    print(f"seed={seed} data_seed={data_seed} mask_seed={mask_seed}")
    print(f"ig_steps={args.ig_steps} baseline={args.baseline} baseline_scalar={args.baseline_scalar}")
    print(f"mask_probs={args.mask_probs} mask_samples_per_prob={args.mask_samples_per_prob} loss_normalize={args.loss_normalize}")
    print(f"ig_postprocess={args.ig_postprocess} mask_batch_size={args.mask_batch_size}")
    print(f"layers={args.layer_start}..{args.layer_end}")
    print(f"use_chat_template={bool(args.use_chat_template)}")
    if str(args.dataset) == "gsm8k":
        print(f"gsm8k_completion_mode={str(args.gsm8k_completion_mode)}")
    print(f"gradient_checkpointing={bool(args.gradient_checkpointing)}")
    print("========================================================")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # Use bf16 weights when requested to reduce memory footprint (especially important for long sequences / large vocab).
    torch_dtype = None
    if bool(args.use_amp_bf16) and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    # Freeze params (we only need grads for alpha)
    for p in model.parameters():
        p.requires_grad_(False)

    if (device.type == "cuda") and (not bool(args.gradient_checkpointing)) and (int(args.max_length) >= 1536):
        print(
            f"[warn] max_length={int(args.max_length)} with gradient_checkpointing=False can be extremely memory-hungry "
            f"for Dream (OOM likely). Consider enabling --gradient_checkpointing or reducing --max_length."
        )

    layers = _find_layers(model)
    num_layers = len(layers)
    num_heads_from_config = int(getattr(getattr(model, "config", None), "num_attention_heads", 0) or 0)
    if num_heads_from_config <= 0:
        raise ValueError("Could not infer num_attention_heads from Dream config.")

    layer_start = int(args.layer_start)
    layer_end = int(args.layer_end)
    if layer_end < 0:
        layer_end = num_layers - 1
    if not (0 <= layer_start <= layer_end < num_layers):
        raise ValueError(
            f"Invalid layer range: {layer_start}..{layer_end} (num_layers={num_layers}). "
            f"Tip: set --layer_end -1 to use the last layer, or use --layer_end {num_layers-1}."
        )

    # Load dataset rows
    if args.dataset == "gsm8k":
        dataset_rows = _load_gsm8k_rows(
            split=str(args.split),
            max_samples=int(args.max_samples),
            seed=int(data_seed),
            dataset_shuffle=bool(args.dataset_shuffle),
        )
    else:
        cats = [c.strip() for c in str(args.nemotron_categories).split(",") if c.strip()]
        dataset_rows = _load_nemotron_rows(
            categories=cats,
            samples_per_category=int(args.samples_per_category),
            pool_per_category=int(args.nemotron_pool_per_category),
            data_seed=int(data_seed),
            max_samples=int(args.max_samples),
        )

    if len(dataset_rows) == 0:
        raise RuntimeError("No dataset rows loaded; please check dataset settings.")

    mask_probs = [float(x.strip()) for x in str(args.mask_probs).split(",") if x.strip()]
    if len(mask_probs) == 0:
        raise ValueError("--mask_probs cannot be empty.")
    for mp in mask_probs:
        if not (0.0 <= mp <= 1.0):
            raise ValueError(f"Invalid mask_prob: {mp}")

    if args.baseline == "zero":
        baseline_value = 0.0
    else:
        baseline_value = float(args.baseline_scalar)

    # Activate checkpointing.
    #
    # IMPORTANT:
    # DreamBaseModel only applies checkpointing when `self.gradient_checkpointing and self.training` is True.
    # We enable it and switch to train() so the checkpoint path is used.
    #
    # To avoid the "inputs must require grad" pitfall and to reduce overhead, we force non-reentrant checkpointing
    # (use_reentrant=False), similar to the LLaDA implementation.
    if bool(args.gradient_checkpointing):
        attn_do = float(getattr(getattr(model, "config", None), "attention_dropout", 0.0) or 0.0)
        preserve_rng_state = bool(attn_do == 0.0)

        # Try HF helper first (newer transformers supports gradient_checkpointing_kwargs).
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # Older transformers: enable default and then override the checkpoint function on the base model.
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        # Ensure base model uses non-reentrant checkpoint.
        if hasattr(model, "model") and hasattr(model.model, "_gradient_checkpointing_func"):
            model.model._gradient_checkpointing_func = partial(
                checkpoint,
                use_reentrant=False,
                preserve_rng_state=preserve_rng_state,
            )
        if hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing"):
            model.model.gradient_checkpointing = True

        if attn_do > 0.0:
            print(
                f"[warn] gradient_checkpointing requires train() in Dream; attention_dropout={attn_do} > 0 "
                f"may introduce stochasticity. Consider setting attention_dropout=0 for determinism."
            )

        model.train()
    else:
        model.eval()

    importance_scores: Dict[int, torch.Tensor] = {}
    for layer_idx in range(layer_start, layer_end + 1):
        imp = compute_layer_ig(
            model,
            layers[layer_idx],
            layer_idx,
            tokenizer,
            dataset_rows,
            device=device,
            ig_steps=int(args.ig_steps),
            baseline_value=float(baseline_value),
            max_length=int(args.max_length),
            num_heads_from_config=int(num_heads_from_config),
            use_amp_bf16=bool(args.use_amp_bf16),
            dataset_name=str(args.dataset),
            dataset_use_chat_template=bool(args.use_chat_template),
            gsm8k_completion_mode=str(args.gsm8k_completion_mode),
            mask_probs=mask_probs,
            mask_samples_per_prob=int(args.mask_samples_per_prob),
            loss_normalize=str(args.loss_normalize),
            seed=int(mask_seed),
            ig_postprocess=str(args.ig_postprocess),
            mask_batch_size=int(args.mask_batch_size),
            show_progress=bool(args.show_progress),
            progress_update_every=int(args.progress_update_every),
            debug_gate=bool(args.debug_gate),
            gradient_checkpointing=bool(args.gradient_checkpointing),
        )
        importance_scores[int(layer_idx)] = imp.detach().cpu()
        print(f"[layer {layer_idx}] mean={imp.mean().item():.6g} std={imp.std(unbiased=False).item():.6g}")

    out = {
        "importance_scores": importance_scores,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model_path": str(args.model_path),
            "dataset": str(args.dataset),
            "split": str(args.split),
            "max_samples": int(args.max_samples),
            "max_length": int(args.max_length),
            "seed": int(seed),
            "data_seed": int(data_seed),
            "mask_seed": int(mask_seed),
            "ig_steps": int(args.ig_steps),
            "baseline": str(args.baseline),
            "baseline_scalar": float(args.baseline_scalar),
            "mask_probs": mask_probs,
            "mask_samples_per_prob": int(args.mask_samples_per_prob),
            "loss_normalize": str(args.loss_normalize),
            "ig_postprocess": str(args.ig_postprocess),
            "mask_batch_size": int(args.mask_batch_size),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "num_layers": int(num_layers),
            "num_attention_heads": int(num_heads_from_config),
            "use_chat_template": bool(args.use_chat_template),
            "gsm8k_completion_mode": str(args.gsm8k_completion_mode) if str(args.dataset) == "gsm8k" else None,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
        },
    }

    out_path = os.path.join(args.output_dir, "head_importance.pt")
    torch.save(out, out_path)
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


