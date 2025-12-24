"""
download_model.py（仅用于下载，下载完成后可删除）

使用 ModelScope 从离线镜像仓库下载 Qwen3-4B 模型权重。
该脚本应在有网络的环境下运行，下载完成后可将模型目录
打包拷贝到评测机的 `/data/local/Qwen3-4B` 等路径下使用。
"""

from modelscope import snapshot_download


def main():
    # ModelScope 仓库名
    # 4bit(bnb) 量化版本（你后续将上传到该 repo）
    model_repo = "SLDRMK/Qwen3-4B-realistic-400-500"

    # snapshot_download 会返回本地模型目录路径
    # 这里将 cache_dir 与 local_dir 显式指定为当前目录下的 Qwen3-4B 目录
    # 这样评测时容器内运行脚本，会在当前工作目录生成 ./Qwen3-4B
    print(f"开始通过 ModelScope 下载模型：{model_repo} 到当前目录 ./Qwen3-4B ...")
    model_dir = snapshot_download(model_repo, cache_dir=".", local_dir="Qwen3-4B")
    print(f"模型已下载到本地目录：{model_dir}")

    # --- 可选：为预量化(bnb4bit)模型补充量化配置（提升不同 transformers 版本的兼容性）---
    # 你的量化导出目录包含 quantize_meta.json 且 safetensors 里大量权重为 U8，并带 quant_state.* 张量。
    # 部分 transformers 版本在缺少 config.quantization_config 时会走“二次量化”路径导致报错。
    try:
        import json
        import os

        local_dir = "Qwen3-4B"
        qm_path = os.path.join(local_dir, "quantize_meta.json")
        cfg_path = os.path.join(local_dir, "config.json")
        if os.path.exists(qm_path) and os.path.exists(cfg_path):
            with open(qm_path, "r", encoding="utf-8") as f:
                qm = json.load(f)
            compute_dtype = str(qm.get("compute_dtype", "float16")).lower()
            qcfg = {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": compute_dtype,
            }

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "quantization_config" not in cfg:
                cfg["quantization_config"] = qcfg
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                print("已为 config.json 写入 quantization_config（bnb4bit），用于提升加载兼容性。")
    except Exception as e:
        print(f"写入 quantization_config 失败（可忽略）：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()