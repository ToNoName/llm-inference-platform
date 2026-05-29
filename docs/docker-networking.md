# Docker 容器访问宿主机网络

## 问题
容器内网关需要 127.0.0.1 访问宿主机的 llama-server
但容器内的localhost是容器自己，无法访问宿主机的服务

## 解决方案
WSL2 + Docker Desktop 环境下，使用 'host.docker.internal' 代替 '127.0.0.1'

## 实现
gateway.py 通过环境变量读取后端地址：
- VLLM_BACKEND_URL
- LLAMA_BACKEND_URL

docker run 时注入：
-e LLAMA_BACKEND_URL="http://host.docker.internal:8002/v1/chat/completions"
