"""
download_model.py（仅用于下载，下载完成后可删除）

使用 ModelScope 从离线镜像仓库下载 Qwen3-4B 模型权重。
该脚本应在有网络的环境下运行，下载完成后可将模型目录
打包拷贝到评测机的 `/data/local/Qwen3-4B` 等路径下使用。
"""

from modelscope import snapshot_download


def main():
    # ModelScope 仓库名
    model_repo = "SLDRMK/Qwen3-4B"

    # snapshot_download 会返回本地模型目录路径
    # 这里将 cache_dir 与 local_dir 显式指定为当前目录下的 Qwen3-4B 目录
    # 这样评测时容器内运行脚本，会在当前工作目录生成 ./Qwen3-4B
    print(f"开始通过 ModelScope 下载模型：{model_repo} 到当前目录 ./Qwen3-4B ...")
    model_dir = snapshot_download(model_repo, cache_dir=".", local_dir="Qwen3-4B")
    print(f"模型已下载到本地目录：{model_dir}")


if __name__ == "__main__":
    main()