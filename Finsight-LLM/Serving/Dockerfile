FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir transformers>=4.40.0 peft>=0.10.0 accelerate>=0.28.0 \
    bitsandbytes>=0.43.0 fastapi>=0.110.0 "uvicorn[standard]>=0.28.0" pydantic>=2.6.0 pyyaml>=6.0.1
COPY data/ ./data/
COPY serving/ ./serving/
COPY configs/ ./configs/
ENV MODEL_DIR=/app/outputs/r16 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1
CMD ["python","serving/app.py","--model_dir","/app/outputs/r16","--host","0.0.0.0","--port","8000"]
