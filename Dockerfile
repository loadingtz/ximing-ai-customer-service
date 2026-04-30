# 熹茗 AI 客服——多阶段构建（约 3GB）
# 第一阶段：装依赖 + 预下载 bge-small-zh 模型（约 95MB）到镜像层缓存
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    XIMING_EMBED_MODEL=BAAI/bge-small-zh-v1.5

WORKDIR /app

# 系统依赖：libgomp1 给 torch 用
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 装 Python 依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 预下载 embedding 模型——build time 一次性，避免运行时拉
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# ──────────────────────────────────────────────
# 第二阶段：拷代码 + 设默认入口
FROM base AS app
COPY . .

EXPOSE 8000

# 默认拉起 Web 服务（FastAPI + 内置 UI）
# 跑 ingestion: docker run ... python -m ingestion.pipeline
# 跑 CLI: docker run -it ... python cli.py
ENTRYPOINT ["python"]
CMD ["-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
