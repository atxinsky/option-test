FROM python:3.11-slim

WORKDIR /app

# 使用阿里云镜像加速
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || true

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 使用清华源安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建数据目录
RUN mkdir -p /app/data /app/configs

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8503

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:8503/_stcore/health || exit 1

# 启动Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8503", "--server.address=0.0.0.0"]
