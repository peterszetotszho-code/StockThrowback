# ---- Frontend build ----
FROM node:20-alpine AS frontend
WORKDIR /build/frontend
# Playwright is only a dev tool for screenshots; skip its browser download.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Backend runtime ----
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Core + API deps, plus the lightweight `openai` client for optional LLM reports.
# Heavy RAG extras (chromadb / sentence-transformers) are intentionally omitted:
# the pipeline falls back to hash embeddings + in-house BM25 + in-memory store.
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt openai

COPY . .
COPY --from=frontend /build/frontend/dist ./frontend/dist

# HF Spaces (and local runs) default to port 7860; Render overrides via $PORT.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
