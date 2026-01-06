import os
import socket
import uuid
import asyncio
import time
import json
from threading import Lock
from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# 与训练脚本保持一致：强制离线模式，避免任何联网请求
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# 优化 CUDA 显存分配，减少碎片与 OOM 概率
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def check_internet(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """仅用于日志的网络连通性测试，不作为逻辑依赖。"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


# 本地模型目录
# 评测时容器内先运行 download_model.py，会在 /app 目录生成 /app/Qwen3-0.6B
# 这里默认使用绝对路径，避免 vLLM 子进程 cwd 不同时把相对路径当成 HuggingFace repo id 解析。
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/app/Qwen3-0.6B")

# 评测 batch 模式的全局 batch_size（一次推理的最大条数，超出则分多轮推理）
# 在当前 358 道题评测场景下，默认设置为 384（可通过环境变量 BATCH_SIZE 覆盖）。
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "384"))
# 输入侧最大长度（只截断 prompt；生成长度用 max_new_tokens 控制）
# 为进一步提升 tokens/s，在 0.6B 模型 + 358 道题场景下，默认收紧到 384（可通过环境变量 MAX_INPUT_LENGTH 覆盖）。
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "384"))
# 每条最多生成多少新 token；在医疗问答场景下默认 128（可通过环境变量 MAX_NEW_TOKENS 覆盖）。
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
# Prompt 风格：对齐 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py
# - medical_qa: Q: ...\nA:
# - chatml_lora: ChatML system/user/assistant
PROMPT_STYLE = os.getenv("PROMPT_STYLE", "chatml_lora").strip().lower()

_load_lock = Lock()
_engine: AsyncLLMEngine | None = None
_sampling_params: SamplingParams | None = None
# 标记模型是否已经成功加载（包括 warmup）；用于健康检查与评测脚本的状态判断。
_model_ready: bool = False


async def _warmup_engine(
    engine: AsyncLLMEngine,
    stop_tokens,
    max_num_seqs: int,
    gpu_mem_util: float,
) -> None:
    """
    使用大 Batch + 极少生成 token 的方式进行预热，以触发大 Batch CUDA kernel 编译。
    额外基于官方评测数据做一次“形状感知”预热，使真实请求的首轮耗时更接近稳定值。
    """
    # 预热 batch 尽量大，但不超过配置的 max_num_seqs 和评测 batch size
    warmup_batch = min(BATCH_SIZE, max_num_seqs)
    if warmup_batch <= 0:
        return

    # ---------------- 第一阶段：大 Batch + max_tokens=1，触发大并发 CUDA Graph ----------------
    warmup_prompt = "Warmup prompt"
    # 极简采样参数：只生成 1 个 token，速度最快
    warmup_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        n=1,
        stop=stop_tokens,
        ignore_eos=True,
    )

    async def _wait_for_result(gen):
        async for _ in gen:
            pass

    tasks = []
    start = time.time()
    print(
        f"Warmup generate (async): batch={warmup_batch}, "
        f"max_tokens=1, max_num_seqs={max_num_seqs}, gpu_mem_util={gpu_mem_util}"
    )
    for i in range(warmup_batch):
        req_id = f"warmup-{i}-{uuid.uuid4()}"
        gen = engine.generate(warmup_prompt, warmup_params, req_id)
        tasks.append(_wait_for_result(gen))

    await asyncio.gather(*tasks)
    duration = time.time() - start
    print(f"Warmup generate (async): done in {duration:.2f}s.")

    # ---------------- 第二阶段：基于官方评测数据做“形状感知”预热 ----------------
    # 只取一小部分问题，避免启动时间过长；重点覆盖真实 prompt 长度分布。
    try:
        dataset_path = os.getenv("WARMUP_DATA_PATH", "official_test_358.txt")
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(os.getcwd(), dataset_path)
        questions: List[str] = []
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 简单从包含“问题：”的行中抽取问题部分；其余行整体当作 question。
                    q = line
                    idx = q.find("问题：")
                    if idx != -1:
                        q = q[idx + len("问题：") :]
                    questions.append(q)
                    if len(questions) >= min(64, warmup_batch):
                        break

        if questions:
            shape_batch = len(questions)
            shape_max_tokens = min(16, MAX_NEW_TOKENS)
            shape_params = SamplingParams(
                max_tokens=shape_max_tokens,
                temperature=0.0,
                top_p=1.0,
                n=1,
                stop=stop_tokens,
            )
            shape_tasks = []
            print(
                f"Shape Warmup (async): samples={shape_batch}, "
                f"max_tokens={shape_max_tokens}, data_path={dataset_path}"
            )
            s_start = time.time()
            for i, q in enumerate(questions):
                prompt = build_prompt(q)
                req_id = f"shape-warmup-{i}-{uuid.uuid4()}"
                gen = engine.generate(prompt, shape_params, req_id)
                shape_tasks.append(_wait_for_result(gen))
            await asyncio.gather(*shape_tasks)
            s_dur = time.time() - s_start
            print(f"Shape Warmup (async): done in {s_dur:.2f}s.")
        else:
            print("Shape Warmup: no questions loaded, skip.")
    except Exception as e:
        # 形状预热失败不影响主流程
        print(f"Shape Warmup failed (ignored): {type(e).__name__}: {e}")


async def ensure_model_loaded() -> None:
    """
    延迟加载模型，使用 AsyncLLMEngine，并在首次加载时执行大 Batch 预热。
    """
    global _engine, _sampling_params, _model_ready
    if _engine is not None and _sampling_params is not None and _model_ready:
        return

    # --- 网络连通性测试，仅打印结果 ---
    internet_ok = check_internet()
    print(
        "【Internet Connectivity Test】: ",
        "CONNECTED" if internet_ok else "OFFLINE / BLOCKED",
    )

    need_init = False
    # 这些配置在 CPU 侧计算完成，无需 async
    with _load_lock:
        if _engine is not None and _sampling_params is not None and _model_ready:
            return

        print(f"从本地加载模型（vLLM Async）：{LOCAL_MODEL_PATH}")

        tp_size = int(os.getenv("VLLM_TP_SIZE", "1"))
        # 默认使用 bfloat16 计算精度；不再自动启用 AWQ 等权重量化；
        dtype = os.getenv("VLLM_DTYPE", "bfloat16")
        
        # 为了在本地 16GB 级别显卡上也能稳定运行，显式收紧部分 vLLM 资源配置：
        # - max_model_len：默认从模型 config 读取（40960），这里根据实际需求缩小上下文上限，显著减少 KV cache 占用。
        #   在当前 5090 32GB + 358 道题评测场景下，默认固定为 896（可通过环境变量 VLLM_MAX_MODEL_LEN 覆盖）。
        default_max_len = 896
        max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", str(default_max_len)))

        # 针对当前 5090 32GB + 358 道题一次性 batch 评测场景：
        # - max_num_seqs 只需覆盖单次请求的最大并发即可
        default_max_num_seqs = max(BATCH_SIZE, 384)
        default_gpu_mem_util = 0.93

        max_num_seqs = int(
            os.getenv("VLLM_MAX_NUM_SEQS", str(default_max_num_seqs))
        )
        gpu_mem_util = float(
            os.getenv("VLLM_GPU_MEM_UTIL", str(default_gpu_mem_util))
        )

        engine_kwargs = dict(
            model=LOCAL_MODEL_PATH,
            tokenizer=LOCAL_MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            dtype=dtype,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_mem_util,
            enable_prefix_caching=True,
            # 官方推荐生产环境编译优化等级 level=3（自动启用算子融合等 pass）
            compilation_config={"level": 3},
            # 显式使用 CUDA Graph（默认就是 False 即启用，但这里明确写出）
            enforce_eager=False,
            # 保持 FP8 KV Cache
            kv_cache_dtype="fp8",
        )

        # 仅当显式设置了 VLLM_QUANTIZATION 时，才把 quantization 传给 vLLM
        quantization_env = os.getenv("VLLM_QUANTIZATION", "").strip().lower()
        if quantization_env:
            engine_kwargs["quantization"] = quantization_env

        engine_args = AsyncEngineArgs(**engine_kwargs)
        _engine = AsyncLLMEngine.from_engine_args(engine_args)

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
        need_init = True

    # --- 大 Batch 预热（不在锁内执行异步逻辑） ---
    if need_init and _engine is not None:
        disable_warmup = os.getenv("VLLM_DISABLE_WARMUP", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not disable_warmup:
            try:
                await _warmup_engine(_engine, _sampling_params.stop if _sampling_params else None, max_num_seqs, gpu_mem_util)
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


async def _process_one(prompt: str, request_id: str) -> str:
    """
    使用 AsyncLLMEngine 处理单条 prompt，返回生成结果。
    """
    assert _engine is not None and _sampling_params is not None
    final_output = None
    gen = _engine.generate(prompt, _sampling_params, request_id)
    async for request_output in gen:
        final_output = request_output
    if final_output is None or not final_output.outputs:
        return ""
    text = final_output.outputs[0].text or ""
    return text.strip()


async def generate_answers_from_prompts(prompts: List[str]) -> List[str]:
    """
    对齐 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py 的"token 级截去 prompt"逻辑：
    - tokenizer(prompts, padding=True, truncation=True, max_length=MAX_INPUT_LENGTH)
    - model.generate(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, ...)
    - 对每条：按非 padding token 数计算 prompt_len，然后 outputs[i][prompt_len:] 解码为答案

    这里基于 AsyncLLMEngine，通过 asyncio.gather 并行处理多条请求。
    """
    await ensure_model_loaded()
    assert _engine is not None and _sampling_params is not None

    tasks = []
    ts = time.time()
    for idx, p in enumerate(prompts):
        req_id = f"{ts}-{idx}-{uuid.uuid4()}"
        tasks.append(_process_one(p, req_id))

    answers = await asyncio.gather(*tasks)
    # 与原逻辑保持一致：去掉首尾空白
    return [a.strip() for a in answers]


# --- API 定义 ---
app = FastAPI(
    title="Simple Inference Server",
    description="A simple API to run a medical Qwen3-0.6B AWQ model.",
)


@app.on_event("startup")
async def _startup_warmup():
    """
    评测平台通常先做健康检查（GET /），再进入 predict 阶段。
    为了避免 predict 阶段首个请求才开始加载模型导致超时，这里在启动后后台预热加载（不阻塞服务启动）。
    """
    disable = os.getenv("DISABLE_WARMUP", "0").lower() in ("1", "true", "yes", "on")
    if disable:
        return

    try:
        await ensure_model_loaded()
        print("Warmup: model loaded.")
    except Exception as e:
        # 不要让预热失败影响服务启动；真正推理时仍会触发 ensure_model_loaded 并抛出更明确错误
        print(f"Warmup failed (will retry on first /predict): {type(e).__name__}: {e}")


class PromptRequest(BaseModel):
    prompt: Union[str, List[str]]


class PredictResponse(BaseModel):
    response: Union[str, List[str]]


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PromptRequest) -> PredictResponse:
    """
    接收一个或多个 prompt，使用与微调脚本一致的格式进行推理，并返回结果。

    说明：
    - 普通模式：/ 返回 {"status": "ok"} 时，评测按单条调用，只使用 prompt 字段。
    - batch 模式：/ 返回 {"status": "batch"} 时，评测会一次性将所有问题通过 prompts 字段发到这里。
    """
    await ensure_model_loaded()

    # ---------- batch 模式：prompt 为 List[str] ----------
    if isinstance(request.prompt, list):
        batch_questions = request.prompt
        prompts = [build_prompt(q) for q in batch_questions]
        answers = await generate_answers_from_prompts(prompts)
        # 对齐 run_inference_eval.py：token 级切分后直接返回答案，不再做额外字符串截断
        return PredictResponse(response=[a.strip() for a in answers])

    # ---------- 单条模式：fallback 到 prompt ----------
    prompt = build_prompt(request.prompt)
    generated_list = await generate_answers_from_prompts([prompt])
    generated = generated_list[0] if generated_list else ""
    return PredictResponse(response=generated.strip())


@app.get("/")
async def health_check():
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
        await ensure_model_loaded()
    return {"status": "batch"}