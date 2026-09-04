# Multi-stage: the build stage carries uv and the compiler toolchain, the
# runtime stage carries neither. A single-stage image ships a compiler to
# production, which is both larger and a wider attack surface than it needs to
# be.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first, in their own layer. Application code changes on every
# commit; the lock file does not, so this layer is reused across builds.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# A non-root user. The process needs to bind a port and read its own code;
# nothing it does requires root, and running as root turns a container escape
# into a host compromise.
RUN useradd --create-home --uid 10001 engine

WORKDIR /app
COPY --from=builder --chown=engine:engine /build/.venv /app/.venv
COPY --from=builder --chown=engine:engine /build/src /app/src
COPY --chown=engine:engine alembic/ /app/alembic/
COPY --chown=engine:engine alembic.ini /app/alembic.ini

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER engine
EXPOSE 8000

# `/health/live` and not `/health/ready`: an orchestrator restarting a pod
# whose database connection dropped will not fix it and may make the outage
# worse by killing in-flight sessions. Readiness is what the load balancer
# asks; liveness is what decides whether to restart.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status == 200 else 1)"

# Migrations are NOT run here. A container that migrates on start races every
# other replica during a rolling deploy, and a failed migration becomes a crash
# loop rather than a deployment that stopped. Run `alembic upgrade head` as its
# own step before the rollout.
CMD ["uvicorn", "evidence_engine.bootstrap.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
