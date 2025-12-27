import os
import socket
from threading import Lock, Thread
from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

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
# 评测时容器内先运行 download_model.py，会在 /app 目录生成 /app/Qwen3-4B
# 这里默认使用绝对路径，避免 vLLM 子进程 cwd 不同时把相对路径当成 HuggingFace repo id 解析。
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/app/Qwen3-4B")

# 评测 batch 模式的全局 batch_size（一次推理的最大条数，超出则分多轮推理）
# 默认 384，与本地评测脚本和 README 中推荐参数保持一致（可通过环境变量 BATCH_SIZE 覆盖）。
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "384"))
# 输入侧最大长度（只截断 prompt；生成长度用 max_new_tokens 控制）
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "1024"))
# 每条最多生成多少新 token（对齐 run_inference_eval.py 默认）
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
# Prompt 风格：对齐 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py
# - medical_qa: Q: ...\nA:
# - chatml_lora: ChatML system/user/assistant
PROMPT_STYLE = os.getenv("PROMPT_STYLE", "chatml_lora").strip().lower()

_load_lock = Lock()
_llm: LLM | None = None
_sampling_params: SamplingParams | None = None
# 标记模型是否已经成功加载（包括轻量 warmup）；用于健康检查与评测脚本的状态判断。
_model_ready: bool = False


def ensure_model_loaded() -> None:
    """
    延迟加载模型，避免容器启动（健康检查阶段）因为加载权重过慢/失败而直接判 Runtime Failed。
    """
    global _llm, _sampling_params, _model_ready
    if _llm is not None and _sampling_params is not None:
        return

    with _load_lock:
        if _llm is not None and _sampling_params is not None:
            return

        # --- 网络连通性测试，仅打印结果 ---
        internet_ok = check_internet()
        print(
            "【Internet Connectivity Test】: ",
            "CONNECTED" if internet_ok else "OFFLINE / BLOCKED",
        )

        # --- 模型加载（完全本地，无网络）---
        print(f"从本地加载模型（vLLM）：{LOCAL_MODEL_PATH}")

        # vLLM 会自动处理 tokenizer 和 prompt/token 的对应关系，
        # 返回结果本身只包含“新生成部分”，无需再手动裁剪 prompt。
        tp_size = int(os.getenv("VLLM_TP_SIZE", "1"))
        dtype = os.getenv("VLLM_DTYPE", "bfloat16")

        # 为了在本地 16GB 级别显卡上也能稳定运行，显式收紧部分 vLLM 资源配置：
        # - max_model_len：默认从模型 config 读取（40960），这里根据实际需求缩小上下文上限，显著减少 KV cache 占用。
        default_max_len = MAX_INPUT_LENGTH + MAX_NEW_TOKENS + 512
        max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", str(default_max_len)))

        # 针对不同显存容量的 GPU，自适应调节 vLLM 的并发与显存占用：
        # - 16GB 左右：保持相对保守（max_num_seqs≈BATCH_SIZE，gpu_memory_util≈0.9）
        # - 24GB 及以上（典型如 RTX 5090）：适当提高并发和显存占用（max_num_seqs 提升到 512，gpu_memory_util≈0.95）
        # 以上只是默认策略，均可通过环境变量 VLLM_MAX_NUM_SEQS / VLLM_GPU_MEM_UTIL 显式覆盖。
        default_max_num_seqs = max(BATCH_SIZE, 384)
        default_gpu_mem_util = 0.92
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total_gb = props.total_memory / (1024**3)
                # 22GB 做一个大致分界：包括 24GB 等中高端卡
                if total_gb >= 22:
                    # 24GB 级别：在保证相对安全的前提下提高并发与显存利用率
                    default_max_num_seqs = max(BATCH_SIZE, 512)
                    default_gpu_mem_util = 0.92
                # 30GB+（如 32GB 5090）：可以进一步加大并发与显存占用来榨干吞吐
                if total_gb >= 30:
                    default_max_num_seqs = max(BATCH_SIZE, 768)
                    default_gpu_mem_util = 0.93
        except Exception:
            # torch 不可用或查询失败时退回到保守默认值
            pass

        max_num_seqs = int(
            os.getenv("VLLM_MAX_NUM_SEQS", str(default_max_num_seqs))
        )
        gpu_mem_util = float(
            os.getenv("VLLM_GPU_MEM_UTIL", str(default_gpu_mem_util))
        )

        _llm = LLM(
            model=LOCAL_MODEL_PATH,
            tokenizer=LOCAL_MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            dtype=dtype,  # "bfloat16" 在 RTX5090 上更友好
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_mem_util,
        )

        stop_tokens = None
        if PROMPT_STYLE == "chatml_lora":
            # ChatML 对话风格：让 vLLM 在 assistant 段结束时自动停掉
            stop_tokens = ["<|im_end|>"]

        _sampling_params = SamplingParams(
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            top_p=1.0,
            n=1,
            stop=stop_tokens,
        )

        # --- 轻量级 warmup：触发一次 generate，提前完成 kernel / cache 初始化 ---
        # 可通过 VLLM_DISABLE_WARMUP=1 关闭（例如在显存特别紧张的调试环境）。
        disable_warmup = os.getenv("VLLM_DISABLE_WARMUP", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not disable_warmup:
            try:
                warmup_batch = min(4, BATCH_SIZE)
                warmup_prompts = ["Warmup prompt"] * warmup_batch
                warmup_max_tokens = min(8, MAX_NEW_TOKENS)
                warmup_params = SamplingParams(
                    max_tokens=warmup_max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                    stop=stop_tokens,
                )
                print(
                    f"Warmup generate: batch={warmup_batch}, max_tokens={warmup_max_tokens}, "
                    f"max_num_seqs={max_num_seqs}, gpu_mem_util={gpu_mem_util}"
                )
                _llm.generate(warmup_prompts, warmup_params)
                print("Warmup generate: done.")
            except Exception as e:
                # 预热失败不应影响正常推理；后续首个真实请求仍会完成初始化。
                print(f"Warmup generate failed (ignored): {type(e).__name__}: {e}")

        # 若执行到此处，视为模型加载流程已完成（即便 warmup 被显式关闭或失败）；
        # 用于 health_check 与评测脚本判断服务是否 ready。
        _model_ready = True


def build_prompt(question: str) -> str:
    """
    与 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py 的 _build_prompt 对齐：
    - medical_qa: Q: <question>\\nA:
    - chatml_lora: 对齐 excluded/Lora_finetune.py 的 ChatML + system/user/assistant 格式
    """
    if PROMPT_STYLE == "chatml_lora":
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        user_content = question
        return f"{system_prompt}<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return f"Q: {question}\nA:"


def generate_answers_from_prompts(prompts: List[str]) -> List[str]:
    """
    对齐 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py 的"token 级截去 prompt"逻辑：
    - tokenizer(prompts, padding=True, truncation=True, max_length=MAX_INPUT_LENGTH)
    - model.generate(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, ...)
    - 对每条：按非 padding token 数计算 prompt_len，然后 outputs[i][prompt_len:] 解码为答案
    """
    ensure_model_loaded()
    assert _llm is not None and _sampling_params is not None

    # vLLM 的 generate 接口本身就是批量化的，返回的 text 只包含“新生成 token”
    outputs = _llm.generate(prompts, _sampling_params)

    answers: List[str] = []
    for out in outputs:
        if not out.outputs:
            answers.append("")
            continue
        text = out.outputs[0].text or ""
        answers.append(text.strip())

    return answers


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
            responses.extend([a.strip() for a in chunk_answers])
        return PredictResponse(response=responses)

    # ---------- 单条模式：fallback 到 prompt ----------
    prompt = build_prompt(request.prompt)
    generated = generate_answers_from_prompts([prompt])[0]
    return PredictResponse(response=generated.strip())


@app.get("/")
def health_check():
    """
    健康检查端点：
    - 返回 {"status": "batch"} 时，评测系统会采用 batch 模式：
      一次性将全部评测问题通过 /predict 的 prompts 字段发送过来。

    评测系统在 docker run - health check 阶段预留了 420s，
    这里选择在 **首次健康检查时就阻塞直到模型加载与预热完成**：
    - 确保 /predict 阶段的时间主要用于真正推理，不被冷启动拖慢；
    - 避免还没 ready 就通过健康检查，导致正式评测首个请求非常慢。

    如需在本地开发时跳过这一行为，可设置环境变量：
    - HEALTHCHECK_SKIP_MODEL=1
    """
    skip_model = os.getenv("HEALTHCHECK_SKIP_MODEL", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not skip_model:
        # 阻塞直到模型加载 + 轻量 warmup 完成；若失败，抛出的异常会让健康检查返回 500。
        ensure_model_loaded()
    return {"status": "batch"}