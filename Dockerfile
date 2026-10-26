FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image small and is enough for local inference
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Optionally bake the models into the image (docker build --build-arg PRELOAD_MODELS=1 .)
ARG PRELOAD_MODELS=0
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG LLM_MODEL=google/flan-t5-base
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')" && \
      python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM as M; AutoTokenizer.from_pretrained('${LLM_MODEL}'); M.from_pretrained('${LLM_MODEL}')"; \
    fi

RUN mkdir -p /app/data /app/chroma_db

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
