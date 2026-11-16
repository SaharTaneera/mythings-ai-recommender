# Python inference API. React demo runs separately with npm.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable \
    && useradd --create-home app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app
USER app
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "mythings_ai_recommender.main:app", "--host", "0.0.0.0", "--port", "8000"]
