#!/usr/bin/env python3
"""
Loss-based head attribution for LLaDA (teacher forcing NLL, answer-only).

This script computes per-layer, per-head importance scores using Integrated Gradients (IG)
over a head-gating vector α_{l,h} inserted right before the attention output projection (o_proj).

Key design choices (as requested):
- Objective: teacher-forcing negative log-likelihood (NLL) on *answer tokens only*
- IG: computed per-layer (one layer at a time)
- Baselines: two options are supported
  - baseline=zero: α starts at 0 and integrates to 1
  - baseline=scalar: α starts at a scalar s (default 0.3) and integrates to 1

Outputs:
- head_importance.pt compatible with the evaluation pipeline:
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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Ensure repo root is on sys.path (so `import models.*` works when running directly)
# File path: adaptive-dllm/models/LLaDA/attribution/loss_attribution/compute_loss_attribution.py
# Go up 4 levels => adaptive-dllm/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.LLaDA.core.modeling import LLaDAModelLM
from models.LLaDA.core.configuration import ActivationCheckpointingStrategy


def _find_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Locate the transformer layers/blocks in the model.
    
    For LLaDA models:
      - model.model.transformer["blocks"] (when block_group_size == 1)
      - model.model.transformer["block_groups"] (when block_group_size > 1)
        where each group is a ModuleList of blocks that needs to be flattened
    """
    # Try common HF layouts
    if hasattr(model, "model"):
        m = model.model
    else:
        m = model

    # LLaDA layout: LLaDAModel has `transformer` ModuleDict with `blocks` or `block_groups`
    if hasattr(m, "transformer"):
        tr = getattr(m, "transformer")
        if isinstance(tr, torch.nn.ModuleDict):
            # Case 1: block_group_size == 1 -> transformer["blocks"] is a ModuleList of blocks
            if "blocks" in tr and isinstance(tr["blocks"], torch.nn.ModuleList):
                return list(tr["blocks"])
            
            # Case 2: block_group_size > 1 -> transformer["block_groups"] is a ModuleList[LLaDABlockGroup]
            # LLaDABlockGroup inherits from ModuleList, so each group is directly iterable
            if "block_groups" in tr and isinstance(tr["block_groups"], torch.nn.ModuleList):
                flat: List[torch.nn.Module] = []
                for group in tr["block_groups"]:
                    # LLaDABlockGroup is a ModuleList subclass containing blocks
                    flat.extend(list(group))
                if len(flat) > 0:
                    return flat

    # Fallback: standard HF layouts
    for attr in ("layers", "h", "decoder_layers"):
        if hasattr(m, attr):
            layers = getattr(m, attr)
            if isinstance(layers, (list, torch.nn.ModuleList)):
                return list(layers)

    # LLaMA-like
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return list(m.model.layers)

    raise AttributeError(
        "Could not locate transformer layers on the model. "
        "Please update `_find_layers` for your model structure."
    )


def _find_attn_and_oproj(layer: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module]:
    # LLaDA blocks: attention output projection is `attn_out` on the block itself.
    if hasattr(layer, "attn_out") and isinstance(getattr(layer, "attn_out"), torch.nn.Module):
        return layer, getattr(layer, "attn_out")

    # HF-style blocks: attention is a submodule and o_proj is inside attention.
    for attn_name in ("self_attn", "attn", "attention"):
        if hasattr(layer, attn_name):
            attn = getattr(layer, attn_name)
            break
    else:
        raise AttributeError(
            "Could not locate attention module/projection on a layer. "
            "Expected either `layer.attn_out` (LLaDA) or one of: self_attn/attn/attention (HF-style)."
        )

    for proj_name in ("o_proj", "out_proj", "wo", "proj"):
        if hasattr(attn, proj_name):
            o_proj = getattr(attn, proj_name)
            if isinstance(o_proj, torch.nn.Module):
                return attn, o_proj

    raise AttributeError(
        "Could not locate attention output projection on attn "
        "(expected one of: o_proj/out_proj/wo/proj)."
    )


def _get_num_heads(attn: torch.nn.Module, fallback_from_config: int | None = None) -> int:
    # LLaDA block passes itself as `attn` in our helper; prefer its config
    if hasattr(attn, "config") and hasattr(attn.config, "n_heads"):
        v = getattr(attn.config, "n_heads")
        if isinstance(v, int) and v > 0:
            return int(v)
    for name in ("num_heads", "n_heads", "n_head", "heads"):
        if hasattr(attn, name):
            v = getattr(attn, name)
            if isinstance(v, int):
                return v
    if fallback_from_config is not None:
        return int(fallback_from_config)
    raise AttributeError("Could not infer num_heads from attention module.")


def _parse_gsm8k_final_answer(answer_str: str) -> str:
    # GSM8K's 'answer' often contains reasoning and a final answer after '####'.
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    return answer_str.strip()


def _build_gsm8k_prompt_and_answer(question: str, answer: str) -> Tuple[str, str]:
    # Keep prompt minimal and stable.
    prompt = f"Question: {question}\nAnswer:"
    final = _parse_gsm8k_final_answer(answer)
    # Put a leading space so the first answer token is separated from "Answer:".
    completion = f" {final}"
    return prompt, completion


def _prepare_nemotron_prompt(sample: Dict[str, Any]) -> str:
    """
    Nemotron format:
      - input: List[Dict]{role, content} OR str
      - system_prompt: optional str
    """
    prompt_parts: List[str] = []

    # system prompt
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


def _build_nemotron_prompt_and_completion(sample: Dict[str, Any]) -> Tuple[str, str]:
    prompt = _prepare_nemotron_prompt(sample)
    output = sample.get("output", "")
    if not isinstance(output, str):
        output = str(output)
    # Leading space to separate from prompt ending tokenization-wise (harmless if prompt ends with newline)
    completion = output if output.startswith((" ", "\n")) else f" {output}"
    return prompt, completion


def _tokenize_pair(
    tokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    max_length: int,
    *,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns:
      input_ids: (1, L)
      attention_mask: (1, L)
      completion_start: int (index into sequence where completion span starts)
    """
    if tokenizer.pad_token_id is None:
        # LLaMA-like tokenizers often have no pad token by default
        tokenizer.pad_token = tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(completion, add_special_tokens=False).input_ids

    bos = []
    if tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]

    input_ids_list = (bos + prompt_ids + answer_ids)[:max_length]
    # Answer starts after bos + prompt
    completion_start = min(len(bos) + len(prompt_ids), len(input_ids_list))

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask, int(completion_start)


def _masked_ce_answer_only(logits: torch.Tensor, labels: torch.Tensor, *, normalize: str) -> torch.Tensor:
    """
    logits: (B, T, V)
    labels: (B, T) with -100 ignored
    """
    loss_sum = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    if normalize == "sum":
        return loss_sum
    if normalize == "mean_masked":
        denom = (labels != -100).sum().clamp(min=1).to(loss_sum.dtype)
        return loss_sum / denom
    raise ValueError(f"Unsupported normalize mode: {normalize}")


def _masked_ce_answer_only_batch(logits: torch.Tensor, labels: torch.Tensor, *, normalize: str) -> torch.Tensor:
    """
    Batch version:
      logits: (B, T, V)
      labels: (B, T) with -100 ignored

    Returns a scalar that is the mean over batch items, where each item is either:
      - sum loss over its supervised (masked) positions        (normalize='sum')
      - mean loss over its supervised (masked) positions       (normalize='mean_masked')
    """
    b, t, v = logits.shape
    # token-wise loss (B, T)
    tok = F.cross_entropy(
        logits.view(-1, v),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(b, t)
    mask = (labels != -100)
    tok = tok * mask.to(tok.dtype)
    per_sum = tok.sum(dim=1)  # (B,)
    if normalize == "sum":
        return per_sum.mean()
    if normalize == "mean_masked":
        denom = mask.sum(dim=1).clamp(min=1).to(per_sum.dtype)
        per_mean = per_sum / denom
        return per_mean.mean()
    raise ValueError(f"Unsupported normalize mode: {normalize}")


def _stable_int_seed(*parts: int) -> int:
    """
    Deterministic seed mixer (independent of PYTHONHASHSEED).
    Returns an int in [0, 2**31-1] suitable for torch.Generator.manual_seed.
    """
    s = ",".join(str(int(p)) for p in parts).encode("utf-8")
    h = hashlib.sha1(s).digest()
    return int.from_bytes(h[:4], "little", signed=False) & 0x7FFFFFFF


def _build_labels_and_masked_inputs_for_completion_span(
    *,
    full_input_ids: torch.Tensor,
    completion_start: int,
    mask_token_id: int,
    mask_prob: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a full input_ids tensor (1, L) that already contains the *ground-truth*
    completion tokens, randomly mask a subset of completion tokens (diffusion-style).

    - input_ids_masked: same shape as full_input_ids, with some completion positions replaced by mask_token_id
    - labels: (1, L) with -100 everywhere except at masked completion positions where label=original_token_id

    This matches diffusion mask-predictor training more closely than "mask all completion".
    """
    if not (0.0 <= float(mask_prob) <= 1.0):
        raise ValueError(f"mask_prob must be in [0,1]. Got {mask_prob}")
    ids = full_input_ids[0]  # (L,)
    L = int(ids.numel())
    start = int(max(0, min(int(completion_start), L)))
    if start >= L:
        # No completion span (truncated away)
        labels = torch.full_like(full_input_ids, fill_value=-100)
        return full_input_ids, labels

    comp_len = L - start
    # Sample mask positions inside completion span
    # (use torch for determinism; keep on same device as ids)
    r = torch.rand(comp_len, device=ids.device, generator=generator)
    mask = r < float(mask_prob)
    if mask_prob > 0.0 and not bool(mask.any()):
        # Ensure at least one token is supervised to avoid zero-loss rows
        j = int(torch.randint(low=0, high=comp_len, size=(1,), device=ids.device, generator=generator).item())
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
    """
    LLaDA is a mask-predictor; generation code uses mask_id=126336.
    Prefer config/tokenizer if available; fallback to 126336.
    """
    tok_id = getattr(tokenizer, "mask_token_id", None)
    if isinstance(tok_id, int) and tok_id >= 0:
        return int(tok_id)
    cfg_id = getattr(getattr(model, "config", None), "mask_token_id", None)
    if isinstance(cfg_id, int) and cfg_id >= 0:
        return int(cfg_id)
    # Fallback to paper/code default (see `models/LLaDA/generation/generate.py`)
    return 126336


@torch.no_grad()
def _dry_run_check_o_proj_shape(o_proj: torch.nn.Module, hidden_size: int) -> None:
    # Not perfect but catches obvious mismatches
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
    mask_probs: List[float],
    mask_samples_per_prob: int,
    loss_normalize: str,
    seed: int,
    ig_postprocess: str,
    mask_batch_size: int,
    show_progress: bool,
    progress_update_every: int,
    debug_gate: bool = False,
) -> torch.Tensor:
    """
    Returns tensor[n_heads] for this layer.
    """
    attn, o_proj = _find_attn_and_oproj(layer)
    n_heads = _get_num_heads(attn, fallback_from_config=num_heads_from_config)

    # Determine head_dim from o_proj input size
    if not hasattr(o_proj, "in_features"):
        raise AttributeError("o_proj has no in_features; cannot infer head_dim safely.")
    hidden_size = int(o_proj.in_features)
    if hidden_size % n_heads != 0:
        raise ValueError(f"hidden_size={hidden_size} not divisible by n_heads={n_heads}")
    head_dim = hidden_size // n_heads
    _dry_run_check_o_proj_shape(o_proj, hidden_size)

    # ---------------------------------------------------------------------
    # Per-head gating for *this layer only* (explicit version)
    #
    # We attach a forward_pre_hook to THIS layer's `o_proj` module instance.
    # During the forward pass of the model, when execution reaches exactly
    # this `o_proj`, the hook scales the input to `o_proj` per head:
    #
    #   x[..., h, :] <- alpha[h] * x[..., h, :]
    #
    # Here `alpha` is a vector of length `n_heads` with requires_grad=True,
    # so gradients w.r.t. each alpha[h] are independent -> we get per-head
    # attributions even if alpha values are initialized to the same scalar.
    # ---------------------------------------------------------------------

    class _OProjHeadGate:
        def __init__(self, o_proj_module: torch.nn.Module, *, n_heads: int, head_dim: int):
            self.o_proj = o_proj_module
            self.n_heads = int(n_heads)
            self.head_dim = int(head_dim)
            self.alpha: Optional[torch.Tensor] = None
            self._handle = None

        def _pre_hook(self, module, inputs):
            # This hook runs *only* when this exact `o_proj` module is called.
            if module is not self.o_proj:
                # Defensive: shouldn't happen, but avoids accidental reuse.
                return inputs
            alpha = self.alpha
            if alpha is None:
                return inputs
            x = inputs[0]
            # x: (B, T, hidden_size) -> (B, T, n_heads, head_dim)
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
        # Accumulate IG attribution across dataset
        ig_sum = torch.zeros(n_heads, device=device, dtype=torch.float32)
        total_items = 0

        # Straight-line path from baseline -> 1
        # IG attribution per dimension i: (1 - baseline) * avg_t grad_i
        scale = float(1.0 - baseline_value)

        # Optional: one-time sanity check that changing alpha changes the forward.
        # This helps debug cases where `_find_attn_and_oproj` points to a module that
        # is not actually used during `model(...)` forward.
        did_debug_check = False

        mask_token_id = _get_mask_token_id(model, tokenizer)

        iterator = enumerate(dataset_rows)
        if show_progress and tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(dataset_rows),
                desc=f"layer {layer_idx}",
                dynamic_ncols=True,
                leave=False,
            )
        for row_idx, row in iterator:
            if dataset_name == "gsm8k":
                prompt, completion = _build_gsm8k_prompt_and_answer(row["question"], row["answer"])
            elif dataset_name == "nemotron":
                prompt, completion = _build_nemotron_prompt_and_completion(row)
            else:
                raise ValueError(f"Unsupported dataset_name: {dataset_name}")

            # Build full (prompt + completion) sequence once, then create multiple masked variants.
            full_input_ids, attention_mask, completion_start = _tokenize_pair(
                tokenizer,
                prompt,
                completion,
                device=device,
                max_length=max_length,
                mask_token_id=mask_token_id,
            )

            # Prepare masked inputs/labels for multiple mask strengths (time steps) and Monte-Carlo samples.
            masked_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
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
                    # Must have at least one supervised token
                    if (labels != -100).sum().item() <= 0:
                        continue
                    masked_batches.append((input_ids_masked, attention_mask, labels))

            if len(masked_batches) == 0:
                continue

            if debug_gate and (not did_debug_check):
                with torch.no_grad():
                    # alpha=1 -> baseline forward
                    gate.alpha = torch.ones(n_heads, device=device, dtype=torch.float32)
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits1 = model(masked_batches[0][0], attention_mask=masked_batches[0][1]).logits
                    else:
                        logits1 = model(masked_batches[0][0], attention_mask=masked_batches[0][1]).logits

                    # alpha with one head attenuated should change logits (almost always)
                    alpha2 = torch.ones(n_heads, device=device, dtype=torch.float32)
                    alpha2[0] = 0.0
                    gate.alpha = alpha2
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits2 = model(masked_batches[0][0], attention_mask=masked_batches[0][1]).logits
                    else:
                        logits2 = model(masked_batches[0][0], attention_mask=masked_batches[0][1]).logits

                    delta = (logits1.to(torch.float32) - logits2.to(torch.float32)).abs().mean().item()
                    if delta <= 0.0:
                        raise RuntimeError(
                            "Debug gate check failed: changing alpha did not change logits. "
                            "Hook may not be attached to the active o_proj module."
                        )
                    print(f"[debug_gate] layer={layer_idx} mean|Δlogits|={delta:.6g}")
                    did_debug_check = True

            grads_accum = torch.zeros(n_heads, device=device, dtype=torch.float32)

            # Pre-stack variants for efficiency (optionally chunked)
            all_input_ids = torch.cat([x for (x, _, _) in masked_batches], dim=0)
            all_attn = torch.cat([m for (_, m, _) in masked_batches], dim=0)
            all_labels = torch.cat([y for (_, _, y) in masked_batches], dim=0)
            n_variants = int(all_input_ids.size(0))

            chunk = int(mask_batch_size)
            if chunk <= 0:
                chunk = n_variants
            
            for k in range(1, ig_steps + 1):
                t = float(k) / float(ig_steps)
                alpha_val = baseline_value + t * (1.0 - baseline_value)
                # IMPORTANT:
                # alpha is a *vector* (one independent scalar per head). Even if all
                # entries are initialized to the same value, each alpha[h] multiplies
                # a different head slice, so gradients differ and we obtain per-head
                # importance scores.
                alpha = torch.full(
                    (n_heads,),
                    fill_value=alpha_val,
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                gate.alpha = alpha

                # Forward (parameters frozen => grads only for alpha)
                model.zero_grad(set_to_none=True)

                # Weighted mean over all masked variants (avoid chunk-size bias).
                loss_weighted_sum = None
                total_variants = 0
                if use_amp_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        for start in range(0, n_variants, chunk):
                            end = min(start + chunk, n_variants)
                            logits = model(all_input_ids[start:end], attention_mask=all_attn[start:end]).logits
                            l = _masked_ce_answer_only_batch(
                                logits,
                                all_labels[start:end],
                                normalize=loss_normalize,
                            )
                            bs = int(end - start)
                            total_variants += bs
                            lw = l * float(bs)
                            loss_weighted_sum = lw if loss_weighted_sum is None else (loss_weighted_sum + lw)
                else:
                    for start in range(0, n_variants, chunk):
                        end = min(start + chunk, n_variants)
                        logits = model(all_input_ids[start:end], attention_mask=all_attn[start:end]).logits
                        l = _masked_ce_answer_only_batch(
                            logits,
                            all_labels[start:end],
                            normalize=loss_normalize,
                        )
                        bs = int(end - start)
                        total_variants += bs
                        lw = l * float(bs)
                        loss_weighted_sum = lw if loss_weighted_sum is None else (loss_weighted_sum + lw)

                if loss_weighted_sum is None or total_variants <= 0:
                    continue
                loss = loss_weighted_sum / float(total_variants)

                loss.backward()
                # NOTE: alpha is a leaf tensor with requires_grad=True. If the hook is active and logits depend on alpha,
                # alpha.grad should be a tensor (possibly all zeros). grad==None usually indicates the graph didn't connect.
                if alpha.grad is None:
                    raise RuntimeError("alpha.grad is None; hook may not be applied correctly.")
                grads_accum += alpha.grad.detach().to(torch.float32)

            # IG for this row
            ig_row = scale * (grads_accum / float(ig_steps))
            if ig_postprocess == "abs":
                ig_sum += ig_row.abs()
            elif ig_postprocess == "signed":
                ig_sum += ig_row
            elif ig_postprocess == "relu":
                ig_sum += torch.clamp(ig_row, min=0.0)
            else:
                raise ValueError(f"Unsupported ig_postprocess: {ig_postprocess}")
            total_items += 1
            if show_progress and tqdm is None and progress_update_every > 0:
                if (total_items % int(progress_update_every)) == 0:
                    print(f"[progress] layer={layer_idx} processed={total_items}/{len(dataset_rows)}")

        if total_items == 0:
            raise RuntimeError("No valid samples were processed; cannot compute attribution.")

        return ig_sum / float(total_items)
    finally:
        # Cleanup hook (even if an exception occurs)
        gate.remove()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "nemotron"])
    p.add_argument("--dataset_config", type=str, default="main")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument(
        "--dataset_shuffle",
        action="store_true",
        default=False,
        help=(
            "Shuffle the dataset before taking the first max_samples. "
            "For gsm8k this changes the actual sample subset when combined with --data_seed."
        ),
    )
    p.add_argument(
        "--samples_per_category",
        type=int,
        default=50,
        help="Nemotron only: number of samples per category to stream and take (before applying max_samples cap).",
    )
    p.add_argument(
        "--nemotron_pool_per_category",
        type=int,
        default=1000,
        help=(
            "Nemotron only (streaming): read this many samples per category into a pool first, "
            "then shuffle (by --data_seed) and take samples_per_category. "
            "Set > samples_per_category if you want different seeds to produce different subsets."
        ),
    )
    p.add_argument(
        "--nemotron_categories",
        type=str,
        default="code,math,science,chat,safety",
        help="Nemotron only: comma-separated categories/splits to sample from.",
    )
    p.add_argument("--seed", type=int, default=1234, help="Default seed (used if --data_seed/--mask_seed are not set).")
    p.add_argument(
        "--data_seed",
        type=int,
        default=-1,
        help="Seed for dataset subsampling/shuffling. -1 means use --seed.",
    )
    p.add_argument(
        "--mask_seed",
        type=int,
        default=-1,
        help="Seed for random masking (diffusion time steps). -1 means use --seed.",
    )
    p.add_argument("--ig_steps", type=int, default=8)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument(
        "--mask_probs",
        type=str,
        default="1.0",
        help=(
            "Comma-separated mask probabilities applied within the completion span (diffusion time steps). "
            "We compute loss for each prob (and multiple samples if configured) then average. "
            "Default '1.0' reproduces the previous behavior (mask all completion tokens). "
            "Example: '0.15,0.3,0.5,0.7,0.9'."
        ),
    )
    p.add_argument(
        "--mask_samples_per_prob",
        type=int,
        default=1,
        help="Monte-Carlo samples per mask_prob (different random masks). Increase for lower variance.",
    )
    p.add_argument(
        "--loss_normalize",
        type=str,
        default="mean_masked",
        choices=["sum", "mean_masked"],
        help="How to normalize CE inside each masked sample. mean_masked is usually better across different mask_probs.",
    )
    p.add_argument(
        "--ig_postprocess",
        type=str,
        default="abs",
        choices=["abs", "signed", "relu"],
        help=(
            "Post-process IG per-head scores before aggregating across samples. "
            "abs reproduces the previous behavior (non-negative importance). "
            "signed keeps direction; relu keeps only positive contributions."
        ),
    )
    p.add_argument(
        "--mask_batch_size",
        type=int,
        default=1,
        help=(
            "Batch size for evaluating multiple masked variants per sample (for efficiency). "
            "0 means use all variants in one batch (may OOM)."
        ),
    )
    p.add_argument(
        "--activation_checkpointing",
        type=str,
        default="none",
        choices=["none", "whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"],
        help="Enable model activation checkpointing to reduce memory during IG backprop (slower but much lower VRAM).",
    )
    p.add_argument(
        "--no_progress",
        action="store_true",
        default=False,
        help="Disable progress display (tqdm if installed, else periodic prints).",
    )
    p.add_argument(
        "--progress_update_every",
        type=int,
        default=10,
        help="If tqdm is not installed, print a progress line every N processed samples (per layer).",
    )
    p.add_argument("--baseline", type=str, default="zero", choices=["zero", "scalar"])
    p.add_argument("--baseline_scalar", type=float, default=0.3, help="Used when --baseline=scalar")
    p.add_argument("--layer_start", type=int, default=0)
    p.add_argument("--layer_end", type=int, default=-1, help="Inclusive. -1 means last layer.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_amp_bf16", action="store_true", default=True)
    p.add_argument(
        "--debug_gate",
        action="store_true",
        default=False,
        help="Run a one-time sanity check that changing alpha changes logits (first processed sample per layer).",
    )
    args = p.parse_args()

    base_seed = int(args.seed)
    data_seed = base_seed if int(args.data_seed) < 0 else int(args.data_seed)
    mask_seed = base_seed if int(args.mask_seed) < 0 else int(args.mask_seed)

    torch.manual_seed(base_seed)

    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("LLaDA Loss-based Head Attribution (Layer-wise IG)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}/{args.dataset_config} split={args.split} max_samples={args.max_samples}")
    print(f"Seeds: base={base_seed} data_seed={data_seed} mask_seed={mask_seed}")
    if bool(args.dataset_shuffle):
        print("Dataset shuffle: enabled")
    else:
        print("Dataset shuffle: disabled")
    print(f"IG steps: {args.ig_steps}")
    print(f"Baseline: {args.baseline} (scalar={args.baseline_scalar})")
    print(f"Max length: {args.max_length}")
    print(f"Mask probs: {args.mask_probs} (samples/prob={args.mask_samples_per_prob}, loss_normalize={args.loss_normalize})")
    print(f"IG postprocess: {args.ig_postprocess} | mask_batch_size: {args.mask_batch_size}")
    print(f"Activation checkpointing: {args.activation_checkpointing}")
    if bool(args.no_progress):
        print("Progress: disabled")
    else:
        print(f"Progress: {'tqdm' if tqdm is not None else 'print'}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    # Load model/tokenizer
    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # Optional activation checkpointing (helps memory for IG backprop).
    if str(args.activation_checkpointing) != "none":
        strat = ActivationCheckpointingStrategy[str(args.activation_checkpointing)]
        # LLaDAModelLM wraps LLaDAModel at `model.model`
        if hasattr(model, "model") and hasattr(model.model, "set_activation_checkpointing"):
            model.model.set_activation_checkpointing(strat)
        else:
            print("[warn] activation checkpointing requested but model.model.set_activation_checkpointing not found; ignoring.")

    # Freeze params: we only need grads wrt alpha (hook input)
    for p_ in model.parameters():
        p_.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load dataset
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", args.dataset_config, split=args.split)
        if bool(args.dataset_shuffle):
            ds = ds.shuffle(seed=int(data_seed))
        rows = [ds[i] for i in range(min(args.max_samples, len(ds)))]
    elif args.dataset == "nemotron":
        # Stream each category:
        # - read a pool of size `nemotron_pool_per_category`
        # - shuffle by (data_seed, cat_idx)
        # - take `samples_per_category`
        # then cap globally by max_samples.
        cats = [c.strip() for c in args.nemotron_categories.split(",") if c.strip()]
        rows = []
        pool_per_category = max(int(args.samples_per_category), int(args.nemotron_pool_per_category))
        for cat_idx, cat in enumerate(cats):
            print(f"Loading Nemotron split={cat} (streaming)...")
            stream = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=cat, streaming=True)
            buf = []
            for i, sample in enumerate(stream):
                buf.append(sample)
                if len(buf) >= int(pool_per_category):
                    break
            # Shuffle buffer deterministically using torch, with a per-category seed
            if len(buf) > 1:
                g = torch.Generator()
                g.manual_seed(_stable_int_seed(int(data_seed), int(cat_idx)))
                idx = torch.randperm(len(buf), generator=g).tolist()
                buf = [buf[j] for j in idx]
            # Take subset (changes with data_seed if pool_per_category > samples_per_category)
            take_n = min(int(args.samples_per_category), len(buf))
            rows.extend(buf[:take_n])
        # global cap
        if len(rows) > int(args.max_samples):
            rows = rows[: int(args.max_samples)]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    layers = _find_layers(model)
    n_layers = len(layers)
    n_heads = int(getattr(model.config, "n_heads", 0) or getattr(model.config, "num_attention_heads", 0) or 0)
    if n_heads <= 0:
        # Fallback: use first layer attention
        attn0, _ = _find_attn_and_oproj(layers[0])
        n_heads = _get_num_heads(attn0)

    layer_start = max(0, int(args.layer_start))
    layer_end = int(args.layer_end)
    if layer_end < 0:
        layer_end = n_layers - 1
    layer_end = min(layer_end, n_layers - 1)
    if layer_start > layer_end:
        raise ValueError(f"Invalid layer range: {layer_start}..{layer_end} (n_layers={n_layers})")

    baseline_value = 0.0 if args.baseline == "zero" else float(args.baseline_scalar)
    if not (0.0 <= baseline_value <= 1.0):
        raise ValueError(f"baseline_value must be in [0,1]. Got {baseline_value}")

    mask_probs = [float(x.strip()) for x in str(args.mask_probs).split(",") if x.strip()]
    if len(mask_probs) == 0:
        raise ValueError("--mask_probs cannot be empty.")
    for mp in mask_probs:
        if not (0.0 <= float(mp) <= 1.0):
            raise ValueError(f"mask_prob must be in [0,1]. Got {mp}")
    if int(args.mask_samples_per_prob) <= 0:
        raise ValueError("--mask_samples_per_prob must be > 0.")

    # Compute per-layer head scores
    importance_scores: Dict[int, torch.Tensor] = {}
    for l in range(layer_start, layer_end + 1):
        print(f"\n--- Layer {l}/{n_layers - 1} ---")
        scores = compute_layer_ig(
            model=model,
            layer=layers[l],
            layer_idx=l,
            tokenizer=tokenizer,
            dataset_rows=rows,
            device=device,
            ig_steps=args.ig_steps,
            baseline_value=baseline_value,
            max_length=args.max_length,
            num_heads_from_config=n_heads,
            use_amp_bf16=bool(args.use_amp_bf16 and device.type == "cuda"),
            dataset_name=args.dataset,
            mask_probs=mask_probs,
            mask_samples_per_prob=int(args.mask_samples_per_prob),
            loss_normalize=str(args.loss_normalize),
            seed=int(mask_seed),
            ig_postprocess=str(args.ig_postprocess),
            mask_batch_size=int(args.mask_batch_size),
            show_progress=(not bool(args.no_progress)),
            progress_update_every=int(args.progress_update_every),
            debug_gate=bool(args.debug_gate),
        )
        importance_scores[l] = scores.detach().cpu()
        print(f"  head_scores: mean={scores.mean().item():.6f}, std={scores.std().item():.6f}, "
              f"min={scores.min().item():.6f}, max={scores.max().item():.6f}")

    out = {
        "importance_scores": importance_scores,
        "metadata": {
            "method": "layerwise_ig_diffusion_masked_ce_answer_only_multit",
            "model_path": args.model_path,
            "dataset": f"{args.dataset}/{args.dataset_config}" if args.dataset == "gsm8k" else f"{args.dataset}",
            "split": args.split,
            "max_samples": int(args.max_samples),
            "seed": int(base_seed),
            "data_seed": int(data_seed),
            "mask_seed": int(mask_seed),
            "dataset_shuffle": bool(args.dataset_shuffle),
            "nemotron_pool_per_category": int(args.nemotron_pool_per_category) if args.dataset == "nemotron" else None,
            "ig_steps": int(args.ig_steps),
            "mask_probs": mask_probs,
            "mask_samples_per_prob": int(args.mask_samples_per_prob),
            "loss_normalize": str(args.loss_normalize),
            "ig_postprocess": str(args.ig_postprocess),
            "mask_batch_size": int(args.mask_batch_size),
            "baseline": args.baseline,
            "baseline_value": float(baseline_value),
            "layer_range": [int(layer_start), int(layer_end)],
            "generated_at": datetime.now().isoformat(),
            "note": (
                "Per-layer IG on head gates α inserted at attention o_proj input. "
                "Objective matches LLaDA's mask-predictor behavior: within the completion/answer span, randomly mask a subset "
                "of tokens with mask_token_id at multiple mask probabilities (diffusion time steps) and average the loss. "
                "Cross-entropy is computed ONLY on masked completion/answer positions (labels != -100). "
                "For GSM8K, completion is the final answer after '####'. For Nemotron, completion is sample['output']."
            ),
        },
    }

    out_path = os.path.join(args.output_dir, "head_importance.pt")
    torch.save(out, out_path)
    print(f"\n✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


