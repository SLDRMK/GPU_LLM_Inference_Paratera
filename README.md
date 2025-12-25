# 大模型推理服务模板(并行科技)

本项目是一个极简的大模型推理服务模板，旨在帮助您快速构建一个可以通过API调用的推理服务器。


## 项目结构

- `Dockerfile`: 用于构建容器镜像的配置文件。**请不要修改此文件的 EXPOSE 端口和 CMD 命令，千万不要添加未经允许的镜像，会把硬盘撑爆**。
- `serve.py`: 推理服务的核心代码。您需要在此文件中修改和优化您的模型加载与推理逻辑。这个程序不能访问Internet。
- `requirements.txt`: Python依赖列表。您可以添加您需要的库。
- `.gitignore`: Git版本控制忽略的文件列表。
- `download_model.py`: 下载权重的脚本，可以自行修改，请确保中国大陆的网络能够下载到。可以把权重托管在阿里云对象存储等云平台，或者参考沐曦模板代码中的托管方式。
- `README.md`: 本说明文档。

## 如何修改

您需要关注的核心文件是 `serve.py`。

目前，它使用 `transformers` 库加载了模型 `Qwen/Qwen2.5-0.5B`。您可以完全替换 `serve.py` 的内容，只要保证容器运行后，能提供模板中的'/predict'和'/'等端点即可。


**重要**: 评测系统会向 `/predict` 端点发送 `POST` 请求，其JSON body格式为：

```json
{
  "prompt": "Your question here"
}

您的服务必须能够正确处理此请求，并返回一个JSON格式的响应，格式为：

```json
{
  "response": "Your model's answer here"
}
```

**请务必保持此API契约不变！**

## 环境说明

### 软件包版本

主要软件包(nvcr.io/nvidia/pytorch:25.04-py3)版本请参考[NGC Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-04.html)


`软件使用的Note`:
- 目前支持

nvcr.io/nvidia/pytorch:25.04-py3 d1eac6220dd9

vllm/vllm-openai:latest 727aad66156b
（该镜像的原始信息为：https://hub.docker.com/layers/vllm/vllm-openai/latest/images/sha256-sha256:6766ce0c459e24b76f3e9ba14ffc0442131ef4248c904efdcbf0d89e38be01fe0

swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.0 d8d39b59e909

- 如果您需要其他的镜像，请参与[问卷](https://tp.wjx.top/vm/OciiNf5.aspx)。

### judge平台的配置说明

judge机器的配置如下：

``` text
os: ubuntu24.04
cpu: 14核
内存: 120GB
磁盘: 492GB（已用72GB）
GPU: RTX5090(显存：32GB)
网络带宽：100Mbps，这个网络延迟的波动性比较大，所以给build阶段预留了25分钟的时间
```

judge系统的配置如下：

``` text
docker build stage: 1500s
docker run - health check stage: 420s
docker run - predict stage: 360s
```

## 本地环境搭建与运行（参考）

### 1) 安装 Docker（Ubuntu 22.04/24.04）

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

> 说明：如果不想每次都 `sudo docker ...`，可将当前用户加入 `docker` 组后重新登录：
>
> ```bash
> sudo usermod -aG docker "$USER"
> ```

### 2) 让容器支持 NVIDIA GPU（安装 NVIDIA Container Toolkit）

若运行容器时出现 “NVIDIA Driver was not detected / GPU functionality will not be available”，通常是未安装或未配置 NVIDIA Container Toolkit。

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证 GPU 是否可被容器使用：

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 3) 构建镜像

在项目根目录执行：

```bash
sudo docker build -t paratera-demo:latest .
```

> 说明：
> - 默认构建会执行 `pip install -r requirements.txt` 和 `python3 download_model.py`，保证容器内依赖与模型齐全。
> - 若仅在本地反复调试代码、且之前已经成功完整构建过一次，可以通过构建参数 **跳过安装与下载** 来加速：
>
> ```bash
> sudo docker build -t paratera-demo:latest . --build-arg SKIP_SETUP=1
> ```
>
> - 评测平台上请使用默认构建方式（不要传 `SKIP_SETUP=1`），否则镜像内可能没有模型和依赖。

**重新构建镜像**（当代码或依赖有更新时）：

如果需要完全重新构建（不使用缓存），可以执行：

```bash
# 停止并删除正在运行的容器（如果有）
sudo docker ps -a | grep paratera-demo | awk '{print $1}' | xargs -r sudo docker stop
sudo docker ps -a | grep paratera-demo | awk '{print $1}' | xargs -r sudo docker rm

# 删除旧镜像（可选）
sudo docker rmi paratera-demo:latest

# 重新构建镜像（不使用缓存）
sudo docker build --no-cache -t paratera-demo:latest .
```

或者简单方式（仅重新构建，保留旧镜像）：

```bash
sudo docker build -t paratera-demo:latest .
```

### 4) 启动容器（CPU / GPU）

- CPU 运行：

```bash
sudo docker run --rm -p 8000:8000 paratera-demo:latest
```

- GPU 运行（推荐参数）：

```bash
sudo docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 paratera-demo:latest
```

## 评测平台（judge）等价运行指令（默认端口 8000）

评测平台会先对 `GET /` 做健康检查，然后用 `POST /predict` 做推理。

建议用下面这条命令本地“等价模拟评测运行”（默认端口 **8000**）：

```bash
sudo docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PROMPT_STYLE=chatml_lora \
  -e BATCH_SIZE=4 \
  -e MAX_INPUT_LENGTH=1024 \
  -e MAX_NEW_TOKENS=200 \
  -p 8000:8000 \
  paratera-demo:latest
```

> 说明：
> - `PROMPT_STYLE` 默认就是 `chatml_lora`，这里显式写出便于确认与评测脚本一致。
> - `BATCH_SIZE` 建议先从 `4` 开始（对齐本地评测脚本默认），避免一次性 batch 过大导致显存 OOM。

### 5) 接口验证（符合评测契约）

健康检查：

```bash
curl -s http://127.0.0.1:8000/ ; echo
```

评测契约：`POST /predict` + `{"prompt":"..."}`，返回 `{"response":"..."}`：

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"你好，简单自我介绍一下"}' ; echo
```

评测 batch 模式契约：`POST /predict` + `{"prompt":[...]}，返回 {"response":[...]}`：

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"prompt":["问题1：你好","问题2：给我一句话总结Transformer","问题3：1+1等于几？"]}' ; echo
```

## 一键评测（official_test.txt * N）：ROUGE-L 与 tokens/s

仓库内提供 `eval_official_http.py`，会解析 `official_test.txt`，复制多遍后以 batch 方式调用 `/predict`，
并计算 **ROUGE-L(F1)** 与 **tokens/s**（tokens/s 需要本地可见的模型目录用于 tokenizer 统计）。

示例（服务已在本机 `8000` 端口运行）：

```bash
conda activate gpu_llm
python eval_official_http.py \
  --no_start_server \
  --server_url http://127.0.0.1:8000 \
  --repeat 1 \
  --model_path /home/sldrmk/WorkSpace/GPU_LLM/out/Qwen3-4B-realistic-bnb4bit-final
```

如果推理时出现 500/显存 OOM，可把 HTTP 请求也再拆小一点：

```bash
python eval_official_http.py \
  --no_start_server \
  --server_url http://127.0.0.1:8000 \
  --repeat 3 \
  --request_chunk_size 4 \
  --model_path /home/sldrmk/WorkSpace/GPU_LLM/out/Qwen3-4B-realistic-bnb4bit-final
```

### 6) 模型文件位置

本模板在镜像构建阶段执行 `python download_model.py`，默认将模型下载到容器内工作目录下：

- 容器内路径：`/app/Qwen3-4B`

如需从镜像导出到宿主机：

```bash
cid=$(docker create paratera-demo:latest)
docker cp "$cid":/app/Qwen3-4B ./Qwen3-4B
docker rm "$cid"
```
