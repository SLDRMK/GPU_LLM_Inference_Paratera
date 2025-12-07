import torch
from fastapi import FastAPI
from pydantic import BaseModel
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket

def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

# 本地模型路径
LOCAL_MODEL_PATH = "./local-model"

# --- 网络连通性测试 ---
internet_ok = check_internet()
print("【Internet Connectivity Test】:",
      "CONNECTED" if internet_ok else "OFFLINE / BLOCKED")

# --- 模型加载（从本地加载，无需网络）---
print(f"从本地加载模型：{LOCAL_MODEL_PATH}")
# 手动加载本地模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH)
# 初始化pipeline（使用本地模型）
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device='cuda'
)
set_seed(42)

# --- API 定义 ---
# 创建FastAPI应用实例
app = FastAPI(
    title="Simple Inference Server",
    description="A simple API to run a small language model."
)

# 定义API请求的数据模型
class PromptRequest(BaseModel):
    prompt: str

# 定义API响应的数据模型
class PredictResponse(BaseModel):
    response: str
    
# --- API 端点 ---
@app.post("/predict", response_model=PredictResponse)
def predict(request: PromptRequest):
    """
    接收一个prompt，使用加载的模型进行推理，并返回结果。
    """

    # raise RuntimeError(request.prompt)

    # 单轮 Prompt
    prompt = f"Q: {request.prompt}\nA:"

    # 使用 max_new_tokens + return_full_text=False 来防止重复 prompt
    model_output = generator(
        prompt,
        max_new_tokens=200,            # 生成长度只限制新增内容
        num_return_sequences=1,
        do_sample=False,              # 关闭采样，稳定输出
        return_full_text=False,       # 只返回新增内容，非常关键！
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = model_output[0]["generated_text"].strip()

    # 截断可能继续生成的 "Q:" 或下一轮问话
    for sep in ["\nQ:", "\nQ ", "Q:", "\nQuestion:", "\n\nQ:"]:
        pos = generated.find(sep)
        if pos != -1:
            generated = generated[:pos].strip()
            break

    # 防止答案开头重复问句
    if generated.startswith(request.prompt):
        generated = generated[len(request.prompt):].strip(" \n:.-")

    return PredictResponse(response=generated)


@app.get("/")
def health_check():
    """
    健康检查端点，用于确认服务是否启动成功。
    """
    return {"status": "ok"}