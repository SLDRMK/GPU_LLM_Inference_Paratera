## 基础镜像：使用评测平台提供的 vLLM OpenAI 镜像（华为云镜像加速）
## 对应 README 中的 swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.0
## 注意：请勿修改 EXPOSE 端口和 CMD 命令
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.0

# 可选构建参数：
# - SKIP_SETUP=1 时跳过 pip 安装和模型下载（本地反复调试时提速）
# - PIP_INDEX_URL 可覆盖默认的 pip 源（默认仍使用清华源，方便评测机）
ARG SKIP_SETUP=0
ARG PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装依赖
# 这一步单独做可以利用Docker的层缓存机制，如果requirements.txt不变，则不会重新安装
COPY requirements.txt .
RUN if [ -n "$PIP_INDEX_URL" ]; then \
      pip config set global.index-url "$PIP_INDEX_URL"; \
    else \
      echo "PIP_INDEX_URL 未设置，使用 pip 默认源"; \
    fi
RUN if [ "$SKIP_SETUP" != "1" ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      echo "SKIP_SETUP=1, skip pip install"; \
    fi

COPY download_model.py . 
RUN if [ "$SKIP_SETUP" != "1" ]; then \
      python3 download_model.py; \
    else \
      echo "SKIP_SETUP=1, skip model download"; \
    fi

# 复制项目中的所有其他文件
COPY . .

# 声明容器对外暴露的端口
EXPOSE 8000

# vllm-openai 基础镜像自带 ENTRYPOINT（api_server.py），这里改为 uvicorn，
# 结合下方 CMD，实际启动命令为：
#   uvicorn serve:app --host 0.0.0.0 --port 8000
ENTRYPOINT ["uvicorn"]

# 容器启动时运行的命令参数（传递给 uvicorn）
CMD ["serve:app", "--host", "0.0.0.0", "--port", "8000"]