import os
import socket
from threading import Lock, Thread
from typing import List, Union

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    set_seed,
)

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


# 本地模型目录
# 评测时容器内先运行 download_model.py，会在当前工作目录生成 ./Qwen3-4B
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./Qwen3-4B")

# 若目录下存在 quantize_meta.json，默认按 bitsandbytes 4bit 方式加载（可用 USE_4BIT=0 强制关闭）
USE_4BIT = os.getenv("USE_4BIT", "auto").lower()  # auto/1/0

# 评测 batch 模式的全局 batch_size（一次推理的最大条数，超出则分多轮推理）
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "384"))

_load_lock = Lock()
_tokenizer = None
_model = None
_generator = None


def _should_use_4bit() -> bool:
    if USE_4BIT in ("0", "false", "no", "off"):
        return False
    if USE_4BIT in ("1", "true", "yes", "on"):
        return True
    # auto
    return os.path.exists(os.path.join(LOCAL_MODEL_PATH, "quantize_meta.json"))


def ensure_model_loaded() -> None:
    """
    延迟加载模型，避免容器启动（健康检查阶段）因为加载权重过慢/失败而直接判 Runtime Failed。
    """
    global _tokenizer, _model, _generator
    if _generator is not None:
        return

    with _load_lock:
        if _generator is not None:
            return

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
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_MODEL_PATH,
                local_files_only=True,
                use_fast=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Fast tokenizer 加载失败，将回退到 slow tokenizer。原因：{type(e).__name__}: {e}")
            _tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_MODEL_PATH,
                local_files_only=True,
                use_fast=False,
                trust_remote_code=True,
            )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        if _should_use_4bit():
            # 从 quantize_meta.json 读取 compute_dtype（缺省用 float16）
            compute_dtype = torch.float16
            try:
                import json

                with open(os.path.join(LOCAL_MODEL_PATH, "quantize_meta.json"), "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if str(meta.get("compute_dtype", "")).lower() in ("bfloat16", "bf16"):
                    compute_dtype = torch.bfloat16
                elif str(meta.get("compute_dtype", "")).lower() in ("float16", "fp16"):
                    compute_dtype = torch.float16
                print(f"检测到 4bit 量化模型，启用 bitsandbytes 4bit 加载（compute_dtype={compute_dtype}）。")
            except Exception as e:
                print(f"读取 quantize_meta.json 失败，将使用默认 compute_dtype=float16。原因：{type(e).__name__}: {e}")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

            # 关键：权重已是“预量化”（Linear 权重 dtype=U8，并带 quant_state/absmax 等张量）。
            # 若走 “量化 on-the-fly”，会把 U8 当成待量化输入触发报错。
            # 规避：把量化配置写进 config，让 transformers 以“已量化 checkpoint”方式装配 bnb 模块并直接加载权重。
            cfg = AutoConfig.from_pretrained(
                LOCAL_MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True,
            )
            try:
                cfg.quantization_config = bnb_config.to_dict()  # type: ignore[attr-defined]
            except Exception:
                cfg.quantization_config = bnb_config  # type: ignore[attr-defined]

            _model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True,
                config=cfg,
                device_map="auto",
                torch_dtype=compute_dtype,
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map=None,
            )
            try:
                _model.to("cuda:0")
            except Exception:
                _model.to("cpu")

        pipeline_kwargs = {
            "task": "text-generation",
            "model": _model,
            "tokenizer": _tokenizer,
        }
        # 若模型是通过 device_map="auto" 加载（accelerate 管理设备），pipeline 不能再传 device 参数
        if not hasattr(_model, "hf_device_map"):
            pipeline_kwargs["device"] = 0 if torch.cuda.is_available() else -1

        _generator = pipeline(**pipeline_kwargs)
        set_seed(42)


def build_medical_prompt(question: str) -> str:
    """
    与 /home/sldrmk/WorkSpace/GPU_LLM/medical_llm_finetune.py 中 build_medical_prompt 对齐：
    形如：Q: <question>\\nA:
    """
    return f"Q: {question}\nA:"


def strip_prompt_from_output(generated_text: str, prompt_text: str, raw_question: str) -> str:
    """
    与 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py 的后处理逻辑对齐：
    - 优先把完整 prompt 从输出里剥离（有些模型/版本会回显 prompt）
    - 再做一次“问句回显”兜底剥离
    """
    text = (generated_text or "").strip()

    # 1) 优先剥离 prompt（只剥离最前面的那一段）
    if prompt_text and text.startswith(prompt_text):
        text = text[len(prompt_text) :].lstrip()
    elif prompt_text and prompt_text in text:
        # 兼容 prompt 不在开头但仍被回显的情况：只移除第一次出现
        idx = text.find(prompt_text)
        if idx != -1:
            text = (text[:idx] + text[idx + len(prompt_text) :]).strip()

    # 2) 兜底：剥离问句本身（避免答案开头重复问句）
    if raw_question and text.startswith(raw_question):
        text = text[len(raw_question) :].lstrip(" \n:.-")

    return text.strip()


# --- API 定义 ---
app = FastAPI(
    title="Simple Inference Server",
    description="A simple API to run a medical Qwen3-4B model.",
)


@app.on_event("startup")
def _startup_warmup():
    """
    评测平台通常先做健康检查（GET /），再进入 predict 阶段。
    为了避免 predict 阶段首个请求才开始加载模型导致超时，这里在启动后后台预热加载（不阻塞服务启动）。
    """
    disable = os.getenv("DISABLE_WARMUP", "0").lower() in ("1", "true", "yes", "on")
    if disable:
        return

    def _bg():
        try:
            ensure_model_loaded()
            print("Warmup: model loaded.")
        except Exception as e:
            # 不要让预热失败影响服务启动；真正推理时仍会触发 ensure_model_loaded 并抛出更明确错误
            print(f"Warmup failed (will retry on first /predict): {type(e).__name__}: {e}")

    Thread(target=_bg, daemon=True).start()


class PromptRequest(BaseModel):
    prompt: Union[str, List[str]]


class PredictResponse(BaseModel):
    response: Union[str, List[str]]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PromptRequest) -> PredictResponse:
    """
    接收一个或多个 prompt，使用与微调脚本一致的格式进行推理，并返回结果。

    说明：
    - 普通模式：/ 返回 {"status": "ok"} 时，评测按单条调用，只使用 prompt 字段。
    - batch 模式：/ 返回 {"status": "batch"} 时，评测会一次性将所有问题通过 prompts 字段发到这里。
    """
    ensure_model_loaded()

    # ---------- batch 模式：prompt 为 List[str] ----------
    if isinstance(request.prompt, list):
        batch_questions = request.prompt
        responses: List[str] = []

        # 分片多轮推理：<=BATCH_SIZE 一轮；否则切片多轮
        for start in range(0, len(batch_questions), BATCH_SIZE):
            chunk_questions = batch_questions[start : start + BATCH_SIZE]
            chunk_prompts = [build_medical_prompt(q) for q in chunk_questions]

            model_outputs = _generator(  # type: ignore[misc]
                chunk_prompts,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=False,
                return_full_text=True,
                pad_token_id=_tokenizer.pad_token_id,  # type: ignore[union-attr]
                eos_token_id=_tokenizer.eos_token_id,  # type: ignore[union-attr]
                batch_size=len(chunk_questions),  # 本轮实际 batch 大小
            )

            for raw_question, prompt_text, outputs in zip(chunk_questions, chunk_prompts, model_outputs):
                generated = outputs[0]["generated_text"].strip()
                generated = strip_prompt_from_output(generated, prompt_text, raw_question)

                # 截断可能继续生成的 "Q:" 或下一轮问话
                for sep in ["\nQ:", "\nQ ", "Q:", "\nQuestion:", "\n\nQ:"]:
                    pos = generated.find(sep)
                    if pos != -1:
                        generated = generated[:pos].strip()
                        break

                responses.append(generated)
        return PredictResponse(response=responses)

    # ---------- 单条模式：fallback 到 prompt ----------
    # 与 /root/medical-llm-finetune/medical_llm_finetune.py 中 build_medical_prompt 完全对齐
    prompt = build_medical_prompt(request.prompt)

    # 使用 max_new_tokens + return_full_text=False 来防止重复 prompt
    model_output = _generator(  # type: ignore[misc]
        prompt,
        max_new_tokens=200,  # 生成长度只限制新增内容
        num_return_sequences=1,
        do_sample=False,  # 关闭采样，稳定输出
        return_full_text=True,
        pad_token_id=_tokenizer.pad_token_id,  # type: ignore[union-attr]
        eos_token_id=_tokenizer.eos_token_id,  # type: ignore[union-attr]
    )

    generated = model_output[0]["generated_text"].strip()
    generated = strip_prompt_from_output(generated, prompt, request.prompt)

    # 截断可能继续生成的 "Q:" 或下一轮问话
    for sep in ["\nQ:", "\nQ ", "Q:", "\nQuestion:", "\n\nQ:"]:
        pos = generated.find(sep)
        if pos != -1:
            generated = generated[:pos].strip()
            break

    return PredictResponse(response=generated)


@app.get("/")
def health_check():
    """
    健康检查端点：
    - 返回 {\"status\": \"batch\"} 时，评测系统会采用 batch 模式：
      一次性将全部评测问题通过 /predict 的 prompts 字段发送过来。
    """
    return {"status": "batch"}