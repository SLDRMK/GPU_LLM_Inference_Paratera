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

目前，它使用 `transformers` 库加载了模型 `Qwen/Qwen2.5-0.5B`。您可以完全替换 `serve.py` 的内容，只要保证容器运行后，能提供模板中的/predict等端点即可。


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
- 目前仅支持nvcr.io/nvidia/pytorch:25.04-py3
- 如果您需要其他的镜像，请参与[问卷](https://tp.wjx.top/vm/OciiNf5.aspx)。

### judge平台的配置说明

judge机器的配置如下：

``` text
os: ubuntu24.04
cpu: 14核
内存: 120GB
磁盘: 50GB
GPU: RTX5090(显存：32GB)
网络带宽：100Mbps，这个网络延迟的波动性比较大，所以给build阶段预留了25分钟的时间
```

judge系统的配置如下：

``` text
docker build stage: 1500s
docker run - health check stage: 180s
docker run - predict stage: 360s
```
