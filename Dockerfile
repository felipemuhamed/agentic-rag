# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS build-base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY app ./app


FROM build-base AS api-builder

ARG LLM_PROVIDER_PACKAGE=langchain-openrouter==0.2.8

RUN python -m pip wheel --wheel-dir /wheels '.[knowledge,rag,web,tracing,monitoring]' "${LLM_PROVIDER_PACKAGE}"


FROM build-base AS knowledge-builder

RUN python -m pip wheel --wheel-dir /wheels '.[knowledge]'


FROM build-base AS rag-builder

ARG LLM_PROVIDER_PACKAGE=langchain-openrouter==0.2.8

RUN python -m pip wheel --wheel-dir /wheels '.[knowledge,rag,web,tracing,monitoring]' "${LLM_PROVIDER_PACKAGE}"


FROM python:${PYTHON_VERSION}-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/chroma /data/checkpoints /data/uploads /home/app/.cache/huggingface \
    && chown -R app:app /data /home/app

WORKDIR /app


FROM runtime-base AS api

ARG LLM_PROVIDER_PACKAGE=langchain-openrouter==0.2.8

RUN --mount=type=bind,from=api-builder,source=/wheels,target=/wheels \
    python -m pip install --no-index --find-links=/wheels \
    'agentic-rag[knowledge,rag,web,tracing,monitoring]' "${LLM_PROVIDER_PACKAGE}"

USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM runtime-base AS knowledge

RUN --mount=type=bind,from=knowledge-builder,source=/wheels,target=/wheels \
    python -m pip install --no-index --find-links=/wheels 'agentic-rag[knowledge]'

USER app

ENTRYPOINT ["agentic-rag-ingest"]


FROM runtime-base AS rag

ARG LLM_PROVIDER_PACKAGE=langchain-openrouter==0.2.8

RUN --mount=type=bind,from=rag-builder,source=/wheels,target=/wheels \
    python -m pip install --no-index --find-links=/wheels \
    'agentic-rag[knowledge,rag,web,tracing,monitoring]' "${LLM_PROVIDER_PACKAGE}"

USER app

ENTRYPOINT ["agentic-rag-ask"]
