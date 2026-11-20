FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install -e ".[ui]"

COPY config ./config
COPY data ./data
COPY apps ./apps
COPY reports ./reports

EXPOSE 8000
CMD ["uvicorn", "amz_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
