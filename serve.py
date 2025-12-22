import os
import socket

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from typing import List, Optional

# 与训练脚本保持一致：强制离线模式，避免任何联网请求
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def check_internet(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """仅用于日志的网络连通性测试，不作为逻辑依赖。"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


# 本地合并后的 Qwen3-4B 模型目录
# 评测时容器内先运行 download_model.py，会在当前工作目录生成 ./Qwen3-4B
LOCAL_MODEL_PATH = "./Qwen3-4B"

# --- 网络连通性测试，仅打印结果 ---
internet_ok = check_internet()
print(
    "【Internet Connectivity Test】: ",
    "CONNECTED" if internet_ok else "OFFLINE / BLOCKED",
)

# --- 模型加载（完全本地，无网络）---
print(f"从本地加载模型：{LOCAL_MODEL_PATH}")

# 加载 tokenizer（仅使用本地文件）
# Qwen 系列在不同 transformers/tokenizers 版本组合下，fast tokenizer 可能因为 tokenizer.json 格式差异而解析失败。
# 这里做一个稳健回退：先尝试 use_fast=True，失败则回退 use_fast=False（需要 tiktoken）。
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True,
    )
except Exception as e:
    print(f"Fast tokenizer 加载失败，将回退到 slow tokenizer。原因：{type(e).__name__}: {e}")
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=True,
    )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型到 GPU
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map=None,
)
try:
    model.to("cuda:0")
except Exception:
    # 若 GPU 不可用，则退回 CPU（评测环境应具备 GPU）
    model.to("cpu")

# 初始化 pipeline（使用本地模型）
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)
set_seed(42)


# --- API 定义 ---
app = FastAPI(
    title="Simple Inference Server",
    description="A simple API to run a medical Qwen3-4B model.",
)


class PromptRequest(BaseModel):
    """
    兼容单条和批量两种调用方式：
    - 单条：传入 prompt 字段
    - 批量：传入 prompts 字段（评测时一次性推送全部问题）
    """

    prompt: Optional[str] = None
    prompts: Optional[List[str]] = None
    batch_size: Optional[int] = 384  # 仅批量时使用，默认 384


class PredictResponse(BaseModel):
    """
    兼容单条和批量两种返回格式：
    - 单条：response
    - 批量：responses
    """

    response: Optional[str] = None
    responses: Optional[List[str]] = None


@app.post("/predict", response_model=PredictResponse)
def predict(request: PromptRequest) -> PredictResponse:
    """
    接收一个或多个 prompt，使用与微调脚本一致的格式进行推理，并返回结果。

    说明：
    - 普通模式：/ 返回 {"status": "ok"} 时，评测按单条调用，只使用 prompt 字段。
    - batch 模式：/ 返回 {"status": "batch"} 时，评测会一次性将所有问题通过 prompts 字段发到这里。
    """
    # ---------- 批量模式：prompts 字段存在 ----------
    if request.prompts:
        prompts = [f"Q: {q}\nA:" for q in request.prompts]
        batch_size = request.batch_size or 384

        model_outputs = generator(
            prompts,
            max_new_tokens=200,
            num_return_sequences=1,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            batch_size=batch_size,
        )

        responses: List[str] = []
        # 对于 text-generation，多输入时返回 List[List[Dict]]
        for raw_question, outputs in zip(request.prompts, model_outputs):
            generated = outputs[0]["generated_text"].strip()

            # 截断可能继续生成的 "Q:" 或下一轮问话
            for sep in ["\nQ:", "\nQ ", "Q:", "\nQuestion:", "\n\nQ:"]:
                pos = generated.find(sep)
                if pos != -1:
                    generated = generated[:pos].strip()
                    break

            # 防止答案开头重复问句
            if generated.startswith(raw_question):
                generated = generated[len(raw_question) :].strip(" \n:.-")

            responses.append(generated)

        return PredictResponse(responses=responses)

    # ---------- 单条模式：fallback 到 prompt ----------
    if not request.prompt:
        # 既没有 prompt 也没有 prompts，返回空响应以避免 500
        return PredictResponse(response="")

    # 与 /root/medical-llm-finetune/medical_llm_finetune.py 中 build_medical_prompt 完全对齐
    prompt = f"Q: {request.prompt}\nA:"

    # 使用 max_new_tokens + return_full_text=False 来防止重复 prompt
    model_output = generator(
        prompt,
        max_new_tokens=200,  # 生成长度只限制新增内容
        num_return_sequences=1,
        do_sample=False,  # 关闭采样，稳定输出
        return_full_text=False,  # 只返回新增内容
        pad_token_id=tokenizer.pad_token_id,
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
    健康检查端点：
    - 返回 {\"status\": \"batch\"} 时，评测系统会采用 batch 模式：
      一次性将全部评测问题通过 /predict 的 prompts 字段发送过来。
    """
    return {"status": "batch"}