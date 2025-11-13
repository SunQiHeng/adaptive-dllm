"""
Example: Block-wise Head Attribution for Diffusion Language Model (LLaDA) - Version 2

每个 block 完成后进行一次归因：
- Block 1: baseline=x_0, actual=x_{s-1}
- Block 2: baseline=x_s, actual=x_{2s-1}
- Block k: baseline=x_{(k-1)s}, actual=x_{ks-1}
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LLaDA.attribution.head_attribution_v2 import BlockwiseIntegratedGradientsAttribution
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


def generate_with_blockwise_attribution(
    model,
    ig_attribution,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
):
    """
    LLaDA 生成过程，每个 block 完成后进行归因。
    
    流程：
    1. 生成 block k
    2. 记录 block 开始状态 (x_{(k-1)*steps_per_block}) 和结束状态 (x_{k*steps_per_block - 1})
    3. 对该 block 进行归因
    4. 继续下一个 block
    
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
    
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((prompt.shape[0], gen_length), 
                      dtype=attention_mask.dtype, device=device)
        ], dim=-1)
    
    # Block-wise generation
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    batch_size = prompt.shape[0]
    
    # Initialize attribution accumulator
    total_attributions = {
        layer_idx: torch.zeros(n_heads, device=device)
        for layer_idx in range(n_layers)
    }
    
    print("\n" + "="*70)
    print("BLOCK-WISE GENERATION WITH ATTRIBUTION")
    print("="*70)
    print(f"Total blocks: {num_blocks}, Steps per block: {steps_per_block}")
    
    # Generation and attribution loop
    for num_block in range(num_blocks):
        print(f"\n{'='*70}")
        print(f"Block {num_block + 1}/{num_blocks}")
        print(f"{'='*70}")
        
        # 记录 block 开始状态
        x_block_start = x.clone()
        
        block_start_pos = prompt.shape[1] + num_block * block_length
        block_end_pos = prompt.shape[1] + (num_block + 1) * block_length
        
        block_mask_index = (x[:, block_start_pos:block_end_pos] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        # Generate this block
        print(f"\n[Phase 1] Generating block {num_block + 1}...")
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
            x0_p[:, block_end_pos:] = -np.inf
            
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
            
            if (i + 1) % 8 == 0 or i == steps_per_block - 1:
                print(f"  Step {i + 1}/{steps_per_block} completed")
        
        # 记录 block 结束状态
        x_block_end = x.clone()
        
        # 该 block 对应的生成位置
        block_positions = list(range(block_start_pos, block_end_pos))
        
        print(f"\n✓ Block {num_block + 1} generation completed")
        print(f"  Block positions: [{block_start_pos}, {block_end_pos})")
        
        # Attribution for this block
        print(f"\n[Phase 2] Computing attribution for block {num_block + 1}...")
        
        for sample_idx in range(batch_size):
            if batch_size > 1:
                print(f"  Sample {sample_idx + 1}/{batch_size}:")
            
            sample_baseline = x_block_start[sample_idx:sample_idx + 1]
            sample_actual = x_block_end[sample_idx:sample_idx + 1]
            sample_attention_mask = (
                attention_mask[sample_idx:sample_idx + 1] if attention_mask is not None else None
            )
            
            # 对每一层计算归因
            for layer_idx in range(n_layers):
                if layer_idx % 8 == 0:
                    layer_range_end = min(layer_idx + 8, n_layers)
                    print(f"    Layers {layer_idx + 1}-{layer_range_end}...", end=" ", flush=True)
                
                attributions = ig_attribution.compute_block_attribution_for_layer(
                    baseline_input_ids=sample_baseline,
                    actual_input_ids=sample_actual,
                    block_positions=block_positions,
                    target_layer_idx=layer_idx,
                    attention_mask=sample_attention_mask
                )
                
                total_attributions[layer_idx] += attributions
                
                if layer_idx % 8 == 7 or layer_idx == n_layers - 1:
                    print("Done")
        
        print(f"\n✓ Block {num_block + 1} attribution completed")
    
    return x, total_attributions


def main():
    print("="*70)
    print("BLOCK-WISE HEAD ATTRIBUTION FOR LLADA")
    print("="*70)
    
    # Load model and tokenizer
    device = 'cuda'
    model_path = "/home/qiheng/Projects/models/LLaDA-8B-Instruct"
    print(f"\n[1/5] Loading model and tokenizer from {model_path}...")
    
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Set padding side to left
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    # Check pad_token_id
    assert tokenizer.pad_token_id != 126336, "Padding ID should not equal mask ID"
    
    # Get model config
    n_layers = model.config.n_layers
    n_heads = model.config.n_heads
    
    print(f"Model loaded: {n_layers} layers, {n_heads} heads per layer")
    
    # Prepare prompt
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
    
    # Tokenize
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
    steps = 128  # 总步数
    
    print(f"\nMax prompt length (padded): {prompt.shape[1]}")
    print(f"Generation length: {gen_length}")
    print(f"Block length: {block_length}")
    print(f"Number of blocks: {gen_length // block_length}")
    print(f"Total steps: {steps}")
    print(f"Steps per block: {steps // (gen_length // block_length)}")
    
    # Initialize attribution
    print("\n[3/5] Initializing Block-wise Integrated Gradients attribution...")
    base_model = model.model if hasattr(model, "model") else model
    ig_attribution = BlockwiseIntegratedGradientsAttribution(
        model=base_model,
        n_steps=10  # IG 的插值步数
    )
    
    # Generate with block-wise attribution
    print("\n[4/5] Generating with block-wise head attribution...")
    
    output, importance_scores = generate_with_blockwise_attribution(
        model=model,
        ig_attribution=ig_attribution,
        prompt=prompt,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        remasking='low_confidence',
        mask_id=126336,
    )
    
    # Decode generated text
    print("\n" + "="*70)
    print("GENERATED TEXT")
    print("="*70)
    
    generated_ids = output[:, prompt.shape[1]:]
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
    
    output_file = "head_importance_blockwise_v2.pt"
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
    
    print("\nThis version (V2):")
    print("✓ Block-wise attribution: 每个 block 完成后归因一次")
    print("✓ Baseline: block 开始时的状态 (x_{block_start})")
    print("✓ Actual: block 结束时的状态 (x_{block_end})")
    print("✓ 输入变化适中，既高效又准确")
    print("✓ 归因目标：该 block 对应位置的 logits")
    print(f"✓ 总共进行 {num_blocks} 次归因 (每个 block 一次)")


if __name__ == "__main__":
    main()

