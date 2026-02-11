FROM python:3.11-slim

WORKDIR /app

# Gradio share 隧道需要这些系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

# Docker 内默认开启公开链接
ENV SHARE=true

CMD ["python", "app.py"]
