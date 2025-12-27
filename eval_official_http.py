import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def http_get_json(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    return json.loads(data)


def http_post_json(url: str, payload: Dict[str, Any], timeout: float = 300.0) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data)
    except HTTPError as e:
        # 评测调试时最关键：把 500 的响应体（通常包含 FastAPI 的 detail）打印出来
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTP {e.code} calling {url}\nResponse body:\n{err_body}") from e


def rouge_l_f1(pred: str, ref: str) -> float:
    """
    完全对齐 /home/sldrmk/WorkSpace/GPU_LLM/run_inference_eval.py 的 ROUGE-L 计算：
    - 使用 jieba.lcut 分词
    - tokens 用空格拼接为字符串
    - rouge_score.rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    - scorer.score(ref_tokens, pred_tokens)["rougeL"].fmeasure
    """
    try:
        import jieba  # type: ignore
        from rouge_score import rouge_scorer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少依赖：jieba / rouge_score。\n"
            "请在当前 python 环境安装：pip install jieba rouge-score\n"
            f"原始错误: {type(e).__name__}: {e}"
        )

    pred_tokens = " ".join(jieba.lcut(pred))
    ref_tokens = " ".join(jieba.lcut(ref))
    if not pred_tokens.strip() or not ref_tokens.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(ref_tokens, pred_tokens)
    return float(score["rougeL"].fmeasure)


def load_tokenizer(model_path: str):
    """
    仅用于本地统计 tokens/s。会尝试 use_fast=True 失败则回退 use_fast=False。
    """
    if not model_path or not os.path.isdir(model_path):
        # 关键：如果本地目录不存在，transformers 会把它当成 hub repo id 校验，导致 HFValidationError。
        # 这里提前返回 None，让上层决定是否用近似统计或提示用户传入正确路径。
        return None
    from transformers import AutoTokenizer  # type: ignore

    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
        )


def count_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        # 近似：按字符数估算（不会因为本地缺模型目录而直接崩）
        return int(len(text))
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return int(len(ids))
    except Exception:
        # fallback：用字符数近似
        return int(len(text))


def parse_official_test(path: str) -> List[Tuple[str, str]]:
    """
    解析 official_test.txt 里形如：
      （1. 问题：... 答案：...
    的 QA 对。
    """
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().replace("\r\n", "\n")

    # 定位每个题块的起点
    starts = [m.start() for m in re.finditer(r"(?:^|\n)\s*[\(（]?\d+\s*[\.、]?\s*问题：", s)]
    if not starts:
        # 兜底：按 “问题：...答案：...” 直接抓取
        pairs = []
        for m in re.finditer(r"问题：(.+?)答案：(.+?)(?=(?:\n\s*[\(（]?\d+\s*[\.、]?\s*问题：)|$)", s, flags=re.S):
            q = m.group(1).strip()
            a = m.group(2).strip()
            if q and a:
                pairs.append((q, a))
        return pairs

    starts.append(len(s))
    pairs: List[Tuple[str, str]] = []
    for i in range(len(starts) - 1):
        block = s[starts[i] : starts[i + 1]].strip()
        qpos = block.find("问题：")
        apos = block.find("答案：")
        if qpos == -1 or apos == -1 or apos <= qpos:
            continue
        q = block[qpos + len("问题：") : apos].strip()
        a = block[apos + len("答案：") :].strip()
        if q and a:
            pairs.append((q, a))
    return pairs


@dataclass
class EvalResult:
    avg_rouge_l: float
    total_tokens: int
    elapsed_s: float
    tokens_per_s: float
    n_samples: int


def run_eval(
    server_url: str,
    qa_pairs: List[Tuple[str, str]],
    repeat: int,
    model_path_for_token_count: str,
    timeout_s: float,
    request_chunk_size: int,
    save_details_path: Optional[str] = None,
) -> EvalResult:
    # 复制三遍（或自定义 repeat）
    qa = qa_pairs * max(1, repeat)
    questions = [q for q, _ in qa]
    refs = [a for _, a in qa]

    tokenizer = load_tokenizer(model_path_for_token_count)
    if tokenizer is None:
        print(
            f"[WARN] 本地 tokenizer 加载失败/模型目录不存在：{model_path_for_token_count}\n"
            f"       将用“字符数”近似 tokens 统计 tokens/s（ROUGE-L 不受影响）。\n"
            f"       若需精确 tokens/s，请把模型目录导出到宿主机，并用 --model_path 指向该目录。"
        )

    # 大批量一次性请求可能导致服务端 OOM/超时；这里支持把请求再按 chunk_size 拆成多次 HTTP 调用。
    t0 = time.perf_counter()
    preds_all: List[str] = []
    if request_chunk_size and request_chunk_size > 0:
        for start in range(0, len(questions), request_chunk_size):
            q_chunk = questions[start : start + request_chunk_size]
            resp = http_post_json(
                f"{server_url.rstrip('/')}/predict",
                payload={"prompt": q_chunk},
                timeout=timeout_s,
            )
            chunk_preds = resp.get("response")
            if not isinstance(chunk_preds, list):
                raise RuntimeError(
                    f"/predict 返回格式不对：期望 response 为 list，实际为：{type(chunk_preds).__name__}"
                )
            preds_all.extend([str(x) if x is not None else "" for x in chunk_preds])
    else:
        resp = http_post_json(
            f"{server_url.rstrip('/')}/predict",
            payload={"prompt": questions},
            timeout=timeout_s,
        )
        preds = resp.get("response")
        if not isinstance(preds, list):
            raise RuntimeError(f"/predict 返回格式不对：期望 response 为 list，实际为：{type(preds).__name__}")
        preds_all = [str(x) if x is not None else "" for x in preds]
    t1 = time.perf_counter()

    # 对齐长度
    if len(preds_all) < len(refs):
        preds_all = preds_all + [""] * (len(refs) - len(preds_all))
    preds_all = preds_all[: len(refs)]

    scores: List[float] = []
    total_tokens = 0
    details = []
    for q, ref, pred_s in zip(questions, refs, preds_all):
        ref_s = str(ref) if ref is not None else ""
        score = rouge_l_f1(pred_s, ref_s)
        scores.append(score)
        total_tokens += count_tokens(tokenizer, pred_s)
        if save_details_path is not None:
            details.append({"question": q, "reference": ref_s, "prediction": pred_s, "rougeL_f1": score})

    if save_details_path is not None:
        os.makedirs(os.path.dirname(save_details_path) or ".", exist_ok=True)
        with open(save_details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

    elapsed = max(1e-9, t1 - t0)
    avg_rouge = sum(scores) / len(scores) if scores else 0.0
    tps = total_tokens / elapsed
    return EvalResult(
        avg_rouge_l=avg_rouge,
        total_tokens=total_tokens,
        elapsed_s=elapsed,
        tokens_per_s=tps,
        n_samples=len(scores),
    )


def wait_server_ready(server_url: str, timeout_s: float = 420.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            info = http_get_json(f"{server_url.rstrip('/')}/", timeout=2.0)
            # 约定：当 / 返回 {"status": "batch"} 或 {"status": "ok"} 时视为 ready；
            # 其它状态（如 initializing）继续轮询。
            status = str(info.get("status", "")).strip().lower()
            if status in ("batch", "ok"):
                return
            last_err = RuntimeError(f"/ 返回状态 {status!r}，尚未 ready")
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"等待服务就绪超时（{timeout_s}s）：{last_err}")


def start_server_subprocess(
    workdir: str,
    host: str,
    port: int,
    env: Dict[str, str],
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "serve:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    # 将 stdout/stderr 直接继承到当前终端，方便看加载/报错
    return subprocess.Popen(cmd, cwd=workdir, env=env)


def main():
    p = argparse.ArgumentParser(description="启动 serve.py 并用 official_test.txt*3 评测 ROUGE-L 与 tokens/s")
    p.add_argument("--workdir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument(
        "--official_test",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "official_test_358.txt",
        ),
    )
    # 默认配置与 README 中推荐的 384 batch 评测参数保持一致
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server_url", type=str, default="")
    p.add_argument(
        "--model_path",
        type=str,
        default=os.getenv(
            "LOCAL_MODEL_PATH", os.path.expanduser("~/data/models/Qwen3-4B")
        ),
        help="用于本地统计 tokens/s 的模型目录（需为宿主机可见的本地路径）。",
    )
    p.add_argument("--timeout", type=float, default=360.0)
    p.add_argument(
        "--request_chunk_size",
        type=int,
        default=384,
        help="把 /predict 的 prompt 列表按该大小拆成多次 HTTP 调用（本地默认 384，对齐 README 中推荐的 BATCH_SIZE）。",
    )

    # ------ 关键 batch / vLLM 参数暴露为命令行选项，方便本地扫点做蒸馏实验 ------
    p.add_argument(
        "--batch_size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "384")),
        help="服务端一次最大处理的样本条数（会通过环境变量 BATCH_SIZE 传给 serve.py）。",
    )
    p.add_argument(
        "--max_input_length",
        type=int,
        default=int(os.getenv("MAX_INPUT_LENGTH", "1024")),
        help="prompt 侧最大长度，会通过 MAX_INPUT_LENGTH 传给 serve.py。",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "256")),
        help="每条样本生成的新 token 上限，会通过 MAX_NEW_TOKENS 传给 serve.py。",
    )
    p.add_argument(
        "--vllm_tp_size",
        type=int,
        default=int(os.getenv("VLLM_TP_SIZE", "1")),
        help="vLLM tensor_parallel_size，单卡 5090 一般保持 1 即可。",
    )
    p.add_argument(
        "--vllm_dtype",
        type=str,
        default=os.getenv("VLLM_DTYPE", "bfloat16"),
        help='vLLM dtype，默认为 "bfloat16"（对 5090 友好）。',
    )
    p.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=int(os.getenv("VLLM_MAX_MODEL_LEN", "0")),
        help=(
            "vLLM 的 max_model_len；0 表示由 serve.py 根据 MAX_INPUT_LENGTH/MAX_NEW_TOKENS 自动推导。"
        ),
    )
    p.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=int(os.getenv("VLLM_MAX_NUM_SEQS", "384")),
        help=(
            "vLLM 的 max_num_seqs；0 表示使用 serve.py 的显存自适应默认值（24GB+ 卡上更激进）。"
        ),
    )
    p.add_argument(
        "--vllm_gpu_mem_util",
        type=float,
        default=float(os.getenv("VLLM_GPU_MEM_UTIL", "0.9")),
        help="vLLM 的 gpu_memory_utilization；0 表示使用 serve.py 的默认策略。",
    )
    p.add_argument(
        "--vllm_disable_warmup",
        action="store_true",
        help=(
            "设置后通过环境变量 VLLM_DISABLE_WARMUP=1 关闭 serve.py 里的轻量 warmup，可避免一次性的预热开销。"
        ),
    )
    p.add_argument("--no_start_server", action="store_true", help="不启动本地 uvicorn，只对已有服务发请求")
    p.add_argument(
        "--details_out",
        type=str,
        default=os.path.join("out", "eval_details_official_x3.json"),
    )
    args = p.parse_args()

    server_url = args.server_url.strip() or f"http://{args.host}:{args.port}"
    qa_pairs = parse_official_test(args.official_test)
    if not qa_pairs:
        raise RuntimeError(f"未能从 {args.official_test} 解析到任何 QA 对")

    proc = None
    if not args.no_start_server:
        env = os.environ.copy()
        # 确保本脚本的设置能传给服务（例如 LOCAL_MODEL_PATH/BATCH_SIZE/MAX_*）
        env.setdefault("LOCAL_MODEL_PATH", args.model_path)

        # ------ 将关键 batch / vLLM 参数通过环境变量传给 serve.py ------
        env["BATCH_SIZE"] = str(args.batch_size)
        env["MAX_INPUT_LENGTH"] = str(args.max_input_length)
        env["MAX_NEW_TOKENS"] = str(args.max_new_tokens)
        env["VLLM_TP_SIZE"] = str(args.vllm_tp_size)
        env["VLLM_DTYPE"] = args.vllm_dtype

        if args.vllm_max_model_len > 0:
            env["VLLM_MAX_MODEL_LEN"] = str(args.vllm_max_model_len)
        if args.vllm_max_num_seqs > 0:
            env["VLLM_MAX_NUM_SEQS"] = str(args.vllm_max_num_seqs)
        if args.vllm_gpu_mem_util > 0:
            env["VLLM_GPU_MEM_UTIL"] = str(args.vllm_gpu_mem_util)
        if args.vllm_disable_warmup:
            env["VLLM_DISABLE_WARMUP"] = "1"

        proc = start_server_subprocess(args.workdir, args.host, args.port, env=env)

    try:
        wait_server_ready(server_url, timeout_s=args.timeout)
        r = run_eval(
            server_url=server_url,
            qa_pairs=qa_pairs,
            repeat=args.repeat,
            model_path_for_token_count=args.model_path,
            timeout_s=args.timeout,
            request_chunk_size=args.request_chunk_size,
            save_details_path=args.details_out,
        )
        print("\n===== Eval Summary =====")
        print(f"samples          : {r.n_samples}")
        print(f"avg ROUGE-L (F1) : {r.avg_rouge_l:.6f}")
        print(f"elapsed (s)      : {r.elapsed_s:.3f}")
        print(f"total tokens     : {r.total_tokens}")
        print(f"tokens/s         : {r.tokens_per_s:.3f}")
        print(f"details saved    : {args.details_out}")
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()


