FROM python:3.11-slim

WORKDIR /app

# 关键！！！换成国内 PyPI 镜像，下载快 100 倍
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY src/gateway/ ./gateway/

EXPOSE 8000

CMD ["uvicorn", "gateway.gateway:app", "--host", "0.0.0.0", "--port", "8000"]