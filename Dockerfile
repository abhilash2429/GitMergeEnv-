FROM python:3.11-slim

WORKDIR /app

ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
ENV HF_TOKEN=""
ENV BASE_URL=http://localhost:7860
ENV PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
COPY models.py /app/models.py
COPY inference.py /app/inference.py
COPY server/ /app/server/

RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
