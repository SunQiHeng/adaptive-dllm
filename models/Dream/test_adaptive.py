#!/usr/bin/env python3
"""
Example script demonstrating how to use Dream with adaptive sparse attention.

This example shows:
1. How to create adaptive sparsity configuration
2. How to load the adaptive model
3. How to generate text with adaptive sparsity
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoConfig


def run_actual_generation(gpu_id=3):
    """
    Example 4: Actually run model generation with one or multiple prompts
    
    Args:
        gpu_id: GPU ID to use (default: 2)
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 4: Actual Model Generation with Prompts")
    print("="*70)
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Model path
    model_path = "/data/qh_models/Dream-v0-Instruct-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n[1] Loading model from: {model_path}")
    print(f"    Using GPU: {gpu_id}")
    print(f"    Device: {device}")
    
    try:
        # Load tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        n_layers = config.num_hidden_layers
        n_heads = config.num_key_value_heads
        print(f"    Layers: {n_layers}, KV Heads: {n_heads}")
        
        # Create adaptive config
        print("\n[2] Creating adaptive sparsity configuration...")
        from models.Dream.sparse.adaptive_utils_dream import create_adaptive_sparsity_config
        
        adaptive_config = create_adaptive_sparsity_config(
            n_layers=n_layers,
            n_heads=n_heads,
            strategy='uniform',
            base_sparsity=0.5,
            min_sparsity=0.1,
            max_sparsity=0.9,
            seed=42
        )
        print(f"    Created adaptive config with base_sparsity=0.5")
        
        # Load model
        print("\n[3] Loading model with adaptive sparse attention...")
        from models.Dream.core.adaptive_sparsed_modeling_dream import AdaptiveDreamModel
        
        model = AdaptiveDreamModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
            adaptive_config=adaptive_config
        )
        model.eval()
        print("    Model loaded successfully!")
        
        # Prepare generation config
        print("\n[4] Preparing generation configuration...")
        from models.Dream.generation_utils.generation_utils_dream import DreamGenerationConfig
        
        generation_config = DreamGenerationConfig(
            max_new_tokens=512,
            steps=512,
            alg='entropy',  # Use entropy like in test_dream.py
            temperature=0.2,  # Use non-zero temperature
            eps=1e-3,
            top_p=0.95,
            alg_temp=0.,
            output_history=True,
            return_dict_in_generate=True,
            mask_token_id=tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else None,
        )
        
        # Sparse parameters
        SparseD_param = {
            'skip': 0.2,
            'select': 0.3,  # Test with select=1.0 as user originally reported
            'block_size': 128,
            'new_generation': 512,
            'whole_steps': 512,
            'adaptive': True,
        }
        
        # Define test prompts
        prompts = [
            "What is the capital of France? please give me a detailed introduction of the city."
        ]
        
        print(f"\n[5] Generating responses for {len(prompts)} prompts...")
        print("="*70)
        
        # Generate for each prompt
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*70}")
            print(f"Prompt {i}/{len(prompts)}:")
            print(f"{'='*70}")
            print(f"Input: {prompt}")
            print(f"{'-'*70}")
            
            # Tokenize using apply_chat_template like test_dream.py
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Generate
            print("Generating...")
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    SparseD_param=SparseD_param
                )
            
            # Decode output like test_dream.py
            generations = [
                tokenizer.decode(g[len(p):].tolist())
                for p, g in zip(input_ids, output.sequences)
            ]
            generated_text = generations[0].split(tokenizer.eos_token)[0]
            print(f"Output: {generated_text}")
            print(f"{'='*70}")
        
        print("\n" + "="*70)
        print("Generation completed for all prompts!")
        print("="*70)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to run generation: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Make sure the model path is correct and you have the required dependencies.")
        return False
    
    return True


def main():
    """Run all examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dream Adaptive Sparse Attention Examples")
    parser.add_argument(
        '--run-generation',
        action='store_true',
        help='Actually run model generation (requires model weights)'
    )
    parser.add_argument(
        '--show-examples',
        action='store_true',
        default=True,
        help='Show example code (default: True)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=3,
        help='GPU ID to use for generation (default: 3)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print("# DREAM ADAPTIVE SPARSE ATTENTION - USAGE EXAMPLES")
    print("#"*70)
    
    success = run_actual_generation(gpu_id=args.gpu)
    if not success:
        print("\n[WARNING] Generation failed. See error messages above.")
    
    print("\n\n" + "#"*70)
    print("# For more details, see:")
    print("#   - README_ADAPTIVE_SPARSE.md")
    print("#   - IMPLEMENTATION_SUMMARY.md")
    print("#   - UPDATES_AND_FIXES.md")
    print("#")
    print("# To run actual generation:")
    print("#   python example_adaptive_sparse_usage.py --run-generation")
    print("#   python example_adaptive_sparse_usage.py --run-generation --gpu 2")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()

