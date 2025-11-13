"""
极端测试：使用全零 att，看是否能改变输出
"""
import torch
import sys
sys.path.insert(0, '/home/qiheng/Projects/adaptive-dllm')

from transformers import AutoTokenizer, AutoModel
from models.LLaDA.attribution.head_attribution import IntegratedGradientsHeadAttribution


def extreme_test():
    print("🧪 Extreme Test: Replace att with zeros")
    print("=" * 60)
    
    device = 'cuda'
    model_path = "/home/qiheng/Projects/models/LLaDA-8B-Instruct"
    
    # 加载模型
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 准备输入
    prompt = "Hello"
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    tokenizer.padding_side = 'left'
    encoded = tokenizer(
        [prompt_text], 
        add_special_tokens=False, 
        padding=True, 
        return_tensors="pt"
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    print(f"✓ Input shape: {input_ids.shape}\n")
    
    # 创建归因对象
    ig_attribution = IntegratedGradientsHeadAttribution(model.model, n_steps=5)
    
    # Test: 使用正常 att vs 全零 att
    target_layer = 15
    
    print(f"Testing layer {target_layer}")
    print("-" * 60)
    
    # 提取正常的 att
    att_normal = ig_attribution._compute_layer_head_att(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_layer_idx=target_layer
    )
    
    # 创建全零 att
    att_zero = torch.zeros_like(att_normal)
    
    print(f"Normal att mean: {att_normal.mean().item():.6f}")
    print(f"Zero att mean:   {att_zero.mean().item():.6f}")
    
    # 使用正常 att
    logits_normal = ig_attribution._forward_with_layer_head_cache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_layer_idx=target_layer,
        head_att_values=att_normal
    )
    
    # 使用全零 att
    logits_zero = ig_attribution._forward_with_layer_head_cache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_layer_idx=target_layer,
        head_att_values=att_zero
    )
    
    print(f"\nLogits with normal att: {logits_normal[0, 0, :5]}")
    print(f"Logits with zero att:   {logits_zero[0, 0, :5]}")
    
    diff = (logits_normal - logits_zero).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nMax difference:  {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff > 0.01:
        print(f"✅ Replacing att with zeros DOES change logits significantly!")
        print(f"   This means _forward_with_layer_head_cache is working!")
        return True
    else:
        print(f"❌ Even replacing att with zeros doesn't change logits much")
        print(f"   Single layer changes have minimal impact on final logits")
        return False


if __name__ == "__main__":
    try:
        success = extreme_test()
        print("\n" + "=" * 60)
        if success:
            print("✅ Test PASSED!")
        else:
            print("⚠️  Single layer changes have minimal impact")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()



