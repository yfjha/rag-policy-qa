FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（sentence-transformers 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY fastapi_rag.py .
COPY app.py .
COPY data/ data/
COPY start.sh .

# 给启动脚本执行权限
RUN chmod +x start.sh

# 暴露两个端口（后端 + 前端）
EXPOSE 8000
EXPOSE 8501

# 启动后端和前端
CMD ["./start.sh"]
