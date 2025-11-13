"""
测试 head_attribution.py 中的自定义 forward 函数是否与官方实现一致
"""
import torch
import sys
sys.path.insert(0, '/home/qiheng/Projects/adaptive-dllm')

from transformers import AutoTokenizer, AutoModel
from models.LLaDA.attribution.head_attribution import IntegratedGradientsHeadAttribution


def test_compute_layer_head_att():
    """
    测试 _compute_layer_head_att 是否与官方模型的输出一致
    
    策略：
    1. 使用官方模型前向传播到目标层，提取 attention 输出
    2. 使用自定义函数计算同一层的 attention 输出
    3. 对比两者是否一致
    """
    print("=" * 80)
    print("Test 1: _compute_layer_head_att")
    print("=" * 80)
    
    device = 'cuda'
    model_path = "/home/qiheng/Projects/models/LLaDA-8B-Instruct"
    
    # 加载模型和 tokenizer
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 准备输入
    prompt = "Hello, how are you?"
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    tokenizer.padding_side = 'left'
    encoded = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input text: {prompt_text[:50]}...")
    
    # 创建归因对象
    ig_attribution = IntegratedGradientsHeadAttribution(model.model, n_steps=10)
    
    # 测试多个层
    test_layers = [0, 15, 31]  # 第一层、中间层、最后一层
    
    for target_layer in test_layers:
        print(f"\n--- Testing Layer {target_layer} ---")
        
        # 方法 1: 使用官方模型 + hook 提取 attention 输出
        import torch.nn.functional as F
        captured_att_official = [None]
        
        original_sdpa = F.scaled_dot_product_attention
        
        def sdpa_with_capture(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            result = original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            captured_att_official[0] = result.clone()
            return result
        
        with torch.no_grad():
            # 获取 embeddings
            x = model.model.transformer.wte(input_ids)
            if model.model.config.input_emb_norm:
                x = x * (model.model.config.d_model ** 0.5)
            
            if not (model.model.config.alibi or model.model.config.rope):
                seq_len = input_ids.shape[1]
                pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
                pos_emb = model.model.transformer.wpe(pos)
                x = pos_emb + x
            
            x = model.model.transformer.emb_drop(x)
            
            # 处理 attention mask and bias (matching official logic)
            attention_mask_input = attention_mask
            if attention_mask_input is not None and 0.0 in attention_mask_input:
                attention_mask_processed = attention_mask_input.to(dtype=torch.float).view(-1, seq_len)[:, None, None, :]
                attention_mask_processed = (1.0 - attention_mask_processed) * torch.finfo(attention_mask_processed.dtype).min
            else:
                attention_mask_processed = None
            
            attention_bias = None
            if attention_mask_processed is not None or model.model.config.alibi:
                if attention_bias is None and not model.model.config.alibi:
                    attention_bias = model.model.get_bidirectional_attention_bias(seq_len, x.device)
                    mask_len = seq_len
                    if attention_mask_processed is not None:
                        mask_len = attention_mask_processed.shape[-1]
                    attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
                    if attention_mask_processed is not None:
                        attention_bias = attention_bias + attention_mask_processed
            
            # 逐层前向传播，使用官方 block
            blocks = model.model.transformer.blocks
            for layer_idx, block in enumerate(blocks):
                if layer_idx == target_layer:
                    # 在目标层，使用 hook 捕获 attention
                    F.scaled_dot_product_attention = sdpa_with_capture
                    try:
                        x, _ = block(x, attention_bias=attention_bias)
                    finally:
                        F.scaled_dot_product_attention = original_sdpa
                    break
                else:
                    # 正常 forward
                    x, _ = block(x, attention_bias=attention_bias)
            
            att_official = captured_att_official[0]
            
            # 继续后面的无用代码（为了保持测试结构）
            if False:
                # Feed-forward
                og_x = x
                x = block.ff_norm(x)
                
                if hasattr(block, 'att_proj'):
                    x = block.ff_proj(x)
                    x = block.act(x)
                    x = block.ff_out(x)
                else:
                    x_gate = block.ff_proj(x)
                    x_up = block.up_proj(x)
                    x = block.act(x_gate) * x_up
                    x = block.ff_out(x)
                
                x = block.dropout(x)
                x = og_x + x
        
        # 方法 2: 使用自定义函数
        att_custom = ig_attribution._compute_layer_head_att(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_layer_idx=target_layer
        )
        
        # 对比结果
        print(f"Official att shape: {att_official.shape}")
        print(f"Custom att shape:   {att_custom.shape}")
        
        # 计算差异
        diff = (att_official - att_custom).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_diff = (diff / (att_official.abs() + 1e-8)).mean().item()
        
        print(f"Max absolute difference:  {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        print(f"Mean relative difference: {relative_diff:.2%}")
        
        # 判断是否通过
        if max_diff < 1e-5 or relative_diff < 0.01:
            print("✅ PASSED: Attention outputs match!")
        else:
            print("❌ FAILED: Attention outputs differ significantly!")
            print(f"Sample official values: {att_official[0, 0, 0, :5]}")
            print(f"Sample custom values:   {att_custom[0, 0, 0, :5]}")


def test_forward_with_layer_head_cache():
    """
    测试 _forward_with_layer_head_cache 是否与官方模型输出一致
    
    策略：
    1. 使用官方模型完整前向传播，得到 logits
    2. 提取某一层的 attention 输出
    3. 使用自定义函数，将该层的 attention 替换为提取的值
    4. 对比两个 logits 是否一致
    """
    print("\n" + "=" * 80)
    print("Test 2: _forward_with_layer_head_cache")
    print("=" * 80)
    
    device = 'cuda'
    model_path = "/home/qiheng/Projects/models/LLaDA-8B-Instruct"
    
    # 加载模型和 tokenizer
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 准备输入
    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    tokenizer.padding_side = 'left'
    encoded = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input text: {prompt_text[:50]}...")
    
    # 创建归因对象
    ig_attribution = IntegratedGradientsHeadAttribution(model.model, n_steps=10)
    
    # 测试多个层
    test_layers = [0, 15, 31]
    
    for target_layer in test_layers:
        print(f"\n--- Testing Layer {target_layer} ---")
        
        # 方法 1: 官方完整前向传播
        with torch.no_grad():
            output_official = model(input_ids, attention_mask=attention_mask)
            logits_official = output_official.logits
        
        # 提取该层的 attention 输出
        with torch.no_grad():
            att_actual = ig_attribution._compute_layer_head_att(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_layer_idx=target_layer
            )
        
        print(f"  Extracted att mean: {att_actual.mean().item():.6f}")
        
        # 方法 2: 使用自定义函数，将该层的 att 替换为提取的值
        logits_custom = ig_attribution._forward_with_layer_head_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_layer_idx=target_layer,
            head_att_values=att_actual
        )
        
        print(f"  Target layer: {target_layer}")
        
        # 对比结果
        print(f"Official logits shape: {logits_official.shape}")
        print(f"Custom logits shape:   {logits_custom.shape}")
    
        # 计算差异
        diff = (logits_official - logits_custom).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_diff = (diff / (logits_official.abs() + 1e-8)).mean().item()
        
        print(f"Max absolute difference:  {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        print(f"Mean relative difference: {relative_diff:.2%}")
        
        # 判断是否通过
        if max_diff < 1e-3 or relative_diff < 0.01:
            print("✅ PASSED: Logits match!")
        else:
            print("❌ FAILED: Logits differ significantly!")
            print(f"Sample official logits: {logits_official[0, 0, :5]}")
            print(f"Sample custom logits:   {logits_custom[0, 0, :5]}")


def test_modified_att_values():
    """
    测试使用修改过的 att 值是否能正确影响输出
    
    策略：
    1. 提取某一层的 attention 输出
    2. 修改该 attention 输出（例如乘以 0.5）
    3. 使用自定义函数前向传播
    4. 验证输出确实发生了变化
    """
    print("\n" + "=" * 80)
    print("Test 3: Modified att values")
    print("=" * 80)
    
    device = 'cuda'
    model_path = "/home/qiheng/Projects/models/LLaDA-8B-Instruct"
    
    # 加载模型和 tokenizer
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 准备输入
    prompt = "The answer is"
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    tokenizer.padding_side = 'left'
    encoded = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # 创建归因对象
    ig_attribution = IntegratedGradientsHeadAttribution(model.model, n_steps=10)
    
    target_layer = 15
    
    # 获取原始 att
    with torch.no_grad():
        att_original = ig_attribution._compute_layer_head_att(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_layer_idx=target_layer
        )
    
    # 原始 logits
    logits_original = ig_attribution._forward_with_layer_head_cache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_layer_idx=target_layer,
        head_att_values=att_original
    )
    
    # 修改 att（乘以不同的系数）
    test_scales = [0.0, 0.5, 2.0]
    
    for scale in test_scales:
        att_modified = att_original * scale
        
        logits_modified = ig_attribution._forward_with_layer_head_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_layer_idx=target_layer,
            head_att_values=att_modified
        )
        
        diff = (logits_original - logits_modified).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nScale: {scale:.1f}")
        print(f"  Max difference:  {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        if scale == 1.0:
            if max_diff < 1e-5:
                print("  ✅ Scale=1.0 produces identical output")
            else:
                print("  ❌ Scale=1.0 should produce identical output!")
        else:
            if max_diff > 1e-3:
                print(f"  ✅ Scale={scale} correctly modifies output")
            else:
                print(f"  ❌ Scale={scale} should modify output more!")


if __name__ == "__main__":
    print("\n" + "🧪" * 40)
    print("Testing head_attribution.py custom forward functions")
    print("🧪" * 40 + "\n")
    
    try:
        # Test 1: _compute_layer_head_att
        test_compute_layer_head_att()
        
        # Test 2: _forward_with_layer_head_cache
        test_forward_with_layer_head_cache()
        
        # Test 3: Modified att values
        test_modified_att_values()
        
        print("\n" + "=" * 80)
        print("✅ All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

