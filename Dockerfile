# Backend API image (Render free plan and local checks). Build context: the repo root,
# because the prompts folder sits next to backend/.
#   docker build -t vantis-api .
#   docker run --rm -p 10000:10000 --env-file backend/.env vantis-api

# Same minor version as the local venv (3.12)
FROM python:3.12-slim

# No .pyc writes at runtime, unbuffered logs (they reach Render's log view at once), smaller pip layers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Packages in their own layer, so a code change does not reinstall them
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Only what the API needs: the app package and the prompts (config.py finds /app/prompts).
# No .env, PDF, tests, scripts or data/ (.dockerignore guards this too).
COPY backend/app backend/app
COPY prompts prompts

# appuser cannot write .pyc files at runtime, so compile now (saves time on every cold start at 0.1 CPU)
RUN python -m compileall -q backend/app

# Run as a non-root user
RUN useradd --create-home --uid 10001 appuser
USER appuser

WORKDIR /app/backend
EXPOSE 10000

# One worker on 0.0.0.0:$PORT (Render sets PORT; 10000 is its default).
# sh -c only expands $PORT; exec makes uvicorn PID 1 so it receives Render's SIGTERM.
# --no-access-log: the app's middleware already logs each request as one JSON line.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --no-access-log"]
