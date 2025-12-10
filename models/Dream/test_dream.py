import os
import argparse
import torch
from transformers import AutoModel, AutoTokenizer

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test Dream model generation")
parser.add_argument('--gpu', type=int, default=2, help='GPU ID to use (default: 2)')
parser.add_argument('--model-path', type=str, default="/data/qh_models/Dream-v0-Instruct-7B", 
                    help='Path to the model (default: local model)')
args = parser.parse_args()

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print(f"Using GPU: {args.gpu}")

model_path = args.model_path
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

messages = [
    {"role": "user", "content": "What is the capital of France? please give me a detailed introduction of the city."}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,  # 减少生成长度
    output_history=True,
    return_dict_in_generate=True,
    steps=32,  # 进一步减少步数以提高质量
    temperature=0.8,  # 稍微提高温度
    top_p=0.9,  # 降低top_p以过滤低概率token
    alg="entropy",
    alg_temp=1.5,  # 提高算法温度增加采样随机性
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0].split(tokenizer.eos_token)[0])