# CPU image for the benchmark client. The mock demo needs no GPU.
# Target `local` adds torch and transformers for scripts/serve_transformers.py.
# That target is large and is not what CI builds.

FROM python:3.11-slim AS bench

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY benchlab ./benchlab
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "benchlab"]
CMD ["run", "--config", "configs/example.yaml"]

FROM bench AS local

RUN pip install --no-cache-dir ".[local]"
ENTRYPOINT ["python"]
CMD ["scripts/serve_transformers.py", "--help"]
