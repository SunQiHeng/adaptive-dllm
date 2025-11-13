"""
Example: Head Attribution for Diffusion Language Model (LLaDA)

将归因过程与真实的 diffusion 生成过程合并，确保归因的位置与实际生成的位置一致。
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用新的模块化导入
from models.LLaDA.attribution import IntegratedGradientsHeadAttribution
from transformers import AutoTokenizer


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, 
        device=mask_index.device, 
        dtype=torch.int64
    ) + base
    
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens


def generate_with_attribution(
    model,
    ig_attribution,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
):
    """
    LLaDA 生成过程，生成完成后进行 head attribution。
    
    流程：
    1. 先完成整个生成过程
    2. 记录初始状态（baseline）和最终状态（actual）
    3. 对每个样本的每一层进行归因
    
    Returns:
        x: 生成的序列
        total_attributions: {layer_idx: tensor(n_heads)} - 累计的 head 重要性
    """
    device = model.device
    n_layers = model.config.n_layers
    n_heads = model.config.n_heads
    
    # Initialize sequence
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), 
        mask_id, 
        dtype=torch.long
    ).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # 保存初始状态（baseline）
    x_baseline = x.clone()
    
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((prompt.shape[0], gen_length), 
                      dtype=attention_mask.dtype, device=device)
        ], dim=-1)
    
    prompt_index = (x != mask_id)
    
    # Block-wise generation
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    batch_size = prompt.shape[0]
    
    # Generation loop (no attribution yet)
    print("\n" + "="*70)
    print("PHASE 1: GENERATION")
    print("="*70)
    
    for num_block in range(num_blocks):
        print(f"\nBlock {num_block + 1}/{num_blocks}")
        
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            # Forward pass
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits
            
            # Add Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)
            
            # Calculate confidence
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # (B, L)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # Mask confidence for non-generation region
            x0_p[:, block_end:] = -np.inf
            
            # Only consider masked positions
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # Select positions to update
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            
            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, i].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            
            # Update x
            x[transfer_index] = x0[transfer_index]
        
        print(f"  ✓ Block {num_block + 1} generation completed")
    
    # 最终生成的序列
    x_final = x.clone()
    
    # 确定生成位置（从 prompt 结束到序列末尾）
    generation_start = prompt.shape[1]
    generation_positions = list(range(generation_start, x.shape[1]))
    
    print(f"\n✓ Generation completed! Generated {len(generation_positions)} tokens per sample")
    
    # Attribution phase
    print("\n" + "="*70)
    print("PHASE 2: ATTRIBUTION")
    print("="*70)
    print(f"Computing attribution for {batch_size} samples...")
    
    # Initialize attribution accumulator
    total_attributions = {
        layer_idx: torch.zeros(n_heads, device=device)
        for layer_idx in range(n_layers)
    }
    
    # 对每个样本进行归因
    for sample_idx in range(batch_size):
        print(f"\n[Sample {sample_idx + 1}/{batch_size}]")
        
        sample_baseline = x_baseline[sample_idx:sample_idx + 1]
        sample_final = x_final[sample_idx:sample_idx + 1]
        sample_attention_mask = (
            attention_mask[sample_idx:sample_idx + 1] if attention_mask is not None else None
        )
        
        # 对每一层计算归因
        for layer_idx in range(n_layers):
            if layer_idx % 8 == 0:
                print(f"  Layers {layer_idx + 1}-{min(layer_idx + 8, n_layers)}...", end=" ", flush=True)
            
            attributions = ig_attribution.compute_head_attribution_for_layer(
                baseline_input_ids=sample_baseline,
                actual_input_ids=sample_final,
                generation_positions=generation_positions,
                target_layer_idx=layer_idx,
                attention_mask=sample_attention_mask
            )
            
            total_attributions[layer_idx] += attributions
            
            if layer_idx % 8 == 7 or layer_idx == n_layers - 1:
                print("Done")
    
    print("\n✓ Attribution completed!")
    
    return x_final, total_attributions


def main():
    print("="*70)
    print("HEAD ATTRIBUTION WITH REAL LLADA GENERATION")
    print("="*70)
    
    # Load model and tokenizer (same way as original_generate.py)
    device = 'cuda'
    model_path = "/home/qiheng/Projects/models/LLaDA-8B-Instruct"
    print(f"\n[1/5] Loading model and tokenizer from {model_path}...")
    
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Set padding side to left (same as original_generate.py)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    # Check pad_token_id (same as original_generate.py)
    assert tokenizer.pad_token_id != 126336, "Padding ID should not equal mask ID"
    
    # Get model config
    n_layers = model.config.n_layers
    n_heads = model.config.n_heads
    
    print(f"Model loaded: {n_layers} layers, {n_heads} heads per layer")
    
    # Prepare prompt (same as original_generate.py)
    print("\n[2/5] Preparing prompts...")
    prompts_text = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
    ]
   
    messages = [{"role": "user", "content": text} for text in prompts_text]
    prompt_texts_formatted = [
        tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
        for message in messages
    ]
    
    # Tokenize with add_special_tokens=False (same as original_generate.py)
    # Note: even for multiple prompts, we need to get attention_mask
    encoded_outputs = tokenizer(
        prompt_texts_formatted,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    prompt = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)
    prompt_lengths = attention_mask.sum(dim=1).tolist()
    
    print(f"Number of prompts: {len(prompts_text)}")
    for idx, (orig_prompt, length) in enumerate(zip(prompts_text, prompt_lengths)):
        print(f"  [Sample {idx}] length: {int(length)} tokens")
        print(f"  [Sample {idx}] prompt: {orig_prompt}")
    
    # Generation parameters
    gen_length = 128  # 生成长度
    block_length = 32  # Block 长度
    steps = 128  # 总步数（会除以 num_blocks）
    
    print(f"Max prompt length (padded): {prompt.shape[1]}")
    print(f"Generation length: {gen_length}")
    print(f"Block length: {block_length}")
    print(f"Total steps: {steps}")
    print(f"Steps per block: {steps // (gen_length // block_length)}")
    
    # Initialize attribution
    print("\n[3/5] Initializing Integrated Gradients attribution...")
    base_model = model.model if hasattr(model, "model") else model
    ig_attribution = IntegratedGradientsHeadAttribution(
        model=base_model,
        n_steps=10  # IG 的插值步数
    )
    
    # Generate with attribution
    print("\n[4/5] Generating with head attribution...")
    print("This will take some time as we compute attribution at each step...")
    
    output, importance_scores = generate_with_attribution(
        model=model,
        ig_attribution=ig_attribution,
        prompt=prompt,
        attention_mask=attention_mask,  # Pass attention_mask (same as original_generate.py)
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,  # No randomness for reproducibility
        remasking='low_confidence',
        mask_id=126336,
    )
    
    # Decode generated text
    print("\n" + "="*70)
    print("GENERATED TEXT")
    print("="*70)
    
    generated_ids = output[:, prompt.shape[1]:]  # Remove prompt padding region
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    print("\nGenerated answers:")
    for idx, (orig_prompt, gen_text) in enumerate(zip(prompts_text, generated_texts)):
        print(f"\n[Sample {idx}] Prompt:")
        print(orig_prompt)
        print(f"\n[Sample {idx}] Answer:")
        print(gen_text)
    
    # Print results
    print("\n" + "="*70)
    print("HEAD IMPORTANCE RESULTS")
    print("="*70)
    
    rankings = ig_attribution.get_head_ranking(importance_scores)
    
    # Show first 3 layers as example
    for layer_idx in range(min(3, n_layers)):
        scores = importance_scores[layer_idx]
        ranking = rankings[layer_idx]
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Mean importance: {scores.mean():.6f}")
        print(f"  Std importance:  {scores.std():.6f}")
        print(f"  Max importance:  {scores.max():.6f} (head {scores.argmax().item()})")
        print(f"  Min importance:  {scores.min():.6f} (head {scores.argmin().item()})")
        
        # Show top-5 and bottom-5
        top_5 = ranking[:5]
        bottom_5 = ranking[-5:]
        print(f"  Top-5 heads:    {top_5}")
        print(f"  Bottom-5 heads: {bottom_5}")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_file = "head_importance_real_generation.pt"
    torch.save({
        'importance_scores': {k: v.cpu() for k, v in importance_scores.items()},
        'rankings': rankings,
        'output': output.cpu(),
        'generated_texts': generated_texts,
        'prompts': prompts_text,
        'formatted_prompts': prompt_texts_formatted,
        'prompt_lengths': prompt_lengths,
        'config': {
            'model_path': model_path,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'gen_length': gen_length,
            'steps': steps,
            'block_length': block_length,
            'n_steps_ig': 10,
        }
    }, output_file)
    
    print(f"\nSaved results to: {output_file}")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    
    print("\nThis version:")
    print("✓ Uses real LLaDA generation logic with real prompt")
    print("✓ Two-phase approach: generation first, then attribution")
    print("✓ Baseline: initial mask state (not zero matrix)")
    print("✓ Actual: final generated state")
    print("✓ Computes attribution using final output logits")
    print("✓ Much more efficient than per-step attribution")
    print("✓ Shows generated text output")


if __name__ == "__main__":
    main()
