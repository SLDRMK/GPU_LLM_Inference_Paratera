# download_model.py（仅用于下载，下载完成后可删除）
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name = "facebook/opt-1.3b"
# 本地保存路径（可自定义，例如 "./local-distilgpt2"）
local_model_path = "./local-model"

# 下载并保存模型和tokenizer
print(f"开始下载模型 {model_name} 到 {local_model_path}...")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
print("下载完成！")