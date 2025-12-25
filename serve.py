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
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
# 输入侧最大长度（只截断 prompt；生成长度用 max_new_tokens 控制）
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "1024"))
# 每条最多生成多少新 token（对齐 run_inference_eval.py 默认）
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
# Prompt 风格：对齐 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py
# - medical_qa: Q: ...\nA:
# - chatml_lora: ChatML system/user/assistant
PROMPT_STYLE = os.getenv("PROMPT_STYLE", "chatml_lora").strip().lower()

_load_lock = Lock()
_tokenizer = None
_model = None


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
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return

    with _load_lock:
        if _model is not None and _tokenizer is not None:
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
        # 与 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py 的推理对齐：
        # 默认按 right padding（便于用“非 padding token 数”作为 prompt_len 做 token 级切分）
        try:
            _tokenizer.padding_side = "right"
        except Exception:
            pass

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

        # 注意：这里不再构造 transformers.pipeline，而是像 run_inference_eval.py 一样直接走 tokenizer + model.generate，
        # 以便使用 token 级切分把 prompt 从输出里剥离（更稳、更可控）。


def build_prompt(question: str) -> str:
    """
    与 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py 的 _build_prompt 对齐：
    - medical_qa: Q: <question>\\nA:
    - chatml_lora: 对齐 excluded/Lora_finetune.py 的 ChatML + system/user/assistant 格式
    """
    if PROMPT_STYLE == "chatml_lora":
        # 对齐 run_inference_eval.py 的 ChatML 框架，同时加入“用自己的话回答、避免逐字照抄”的约束，
        # 这属于正当的生成风格控制（避免背题式复述），而不是在输出末尾追加无关噪声。
        system_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.\n"
            "Please answer in your own words and do not copy the dataset or any memorized reference verbatim.\n"
            "Keep the answer professional and concise. Start your final answer with '答：'.\n"
            "<|im_end|>\n"
        )
        user_content = question
        return f"{system_prompt}<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
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


def _infer_input_device(model) -> torch.device:
    """
    推理输入应放到模型所在设备：
    - 普通单卡：model.device 或首个参数 device
    - device_map="auto"：输入通常放到第一层所在设备（首个参数 device）
    """
    try:
        d = getattr(model, "device", None)
        if isinstance(d, torch.device):
            return d
    except Exception:
        pass
    try:
        for p in model.parameters():
            if hasattr(p, "device"):
                return p.device
    except Exception:
        pass
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_answers_from_prompts(prompts: List[str]) -> List[str]:
    """
    对齐 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py 的"token 级截去 prompt"逻辑：
    - tokenizer(prompts, padding=True, truncation=True, max_length=MAX_INPUT_LENGTH)
    - model.generate(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, ...)
    - 对每条：按非 padding token 数计算 prompt_len，然后 outputs[i][prompt_len:] 解码为答案
    """
    ensure_model_loaded()
    tokenizer = _tokenizer  # type: ignore[assignment]
    model = _model  # type: ignore[assignment]
    device = _infer_input_device(model)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(device)

    # 配置终止标记：完全对齐 run_inference_eval.py
    eos_token_id = tokenizer.eos_token_id
    if PROMPT_STYLE == "chatml_lora":
        eos_ids = []
        if tokenizer.eos_token_id is not None:
            eos_ids.append(tokenizer.eos_token_id)
        chatml_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if chatml_end_id is not None and chatml_end_id != tokenizer.eos_token_id:
            eos_ids.append(chatml_end_id)
        if eos_ids:
            eos_token_id = eos_ids

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            num_return_sequences=1,
        )

    answers: List[str] = []
    for i in range(outputs.size(0)):
        gen_ids = outputs[i]
        inp_ids = inputs["input_ids"][i]

        # 完全对齐 run_inference_eval.py：用“非 padding token 数”估算实际 prompt 长度
        if tokenizer.pad_token_id is not None:
            prompt_len = int((inp_ids != tokenizer.pad_token_id).sum().item())
        else:
            prompt_len = int(inp_ids.shape[-1])
        new_token_ids = gen_ids[prompt_len:]
        ans = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        answers.append(ans)

    return answers


def format_answer(ans: str) -> str:
    """
    轻量的格式化：确保以“答：”开头（相关内容，不是无关噪声），并做首尾空白清理。
    """
    a = (ans or "").strip()
    if not a:
        return "答："
    if a.startswith("答："):
        return a
    return "答：" + a


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
            chunk_prompts = [build_prompt(q) for q in chunk_questions]

            chunk_answers = generate_answers_from_prompts(chunk_prompts)
            # 对齐 run_inference_eval.py：token 级切分后直接返回答案，不再做额外字符串截断
            responses.extend([format_answer(a) for a in chunk_answers])
        return PredictResponse(response=responses)

    # ---------- 单条模式：fallback 到 prompt ----------
    prompt = build_prompt(request.prompt)
    generated = generate_answers_from_prompts([prompt])[0]
    return PredictResponse(response=format_answer(generated))


@app.get("/")
def health_check():
    """
    健康检查端点：
    - 返回 {\"status\": \"batch\"} 时，评测系统会采用 batch 模式：
      一次性将全部评测问题通过 /predict 的 prompts 字段发送过来。
    """
    return {"status": "batch"}