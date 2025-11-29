# 大模型推理服务模板(并行科技)

本项目是一个极简的大模型推理服务模板，旨在帮助您快速构建一个可以通过API调用的推理服务器。


## 项目结构

- `FakeDockerfile`: 用于构建容器镜像的配置文件,但是这个是假的，因为摩尔线程的评测机是容器实例，没有办法再创建容器了，这里的FROM并不是拉取容器镜像。**请不要修改此文件的 EXPOSE 端口和 CMD 命令**。
- `serve.py`: 推理服务的核心代码。您需要在此文件中修改和优化您的模型加载与推理逻辑。这个程序不能访问Internet。
- `requirements.txt`: Python依赖列表。您可以添加您需要的库。
- `.gitignore`: Git版本控制忽略的文件列表。
- `download_model.py`: 下载权重的脚本，可以自行修改，请确保中国大陆的网络能够下载到。
- `README.md`: 本说明文档。

## 如何修改

您需要关注的核心文件是 `serve.py`。

目前，它使用 `transformers` 库加载了一个非常小的模型 `distilgpt2`。您可以：

1.  **替换模型**: 将 `model='distilgpt2'` 替换为您希望使用的其他模型。
2.  **优化推理逻辑**: 在 `predict` 函数中，您可以修改 `generator()` 的参数，如 `max_length`, `top_k`, `top_p` 等，以获得更好的生成效果。
3.  **使用其他推理框架**: 您可以完全替换 `serve.py` 的内容，只要保证容器运行后，能提供一个接收 `POST` 请求的 `/predict` 端点即可。

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
