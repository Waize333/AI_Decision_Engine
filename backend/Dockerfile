# ============================================================
# backend/Dockerfile
# ============================================================
# PURPOSE:
#   Packages the FastAPI application into a production-ready
#   Docker container.
#
# MULTI-STAGE BUILD:
#   Stage 1 (builder) — installs Python dependencies
#   Stage 2 (runtime) — copies only what's needed to run
#
# WHY MULTI-STAGE?
#   The builder stage has compilers, headers, and build tools
#   that aren't needed at runtime. By copying only the final
#   site-packages into a clean image, we get:
#     - Smaller image (~200MB vs ~800MB)
#     - Smaller attack surface (no gcc, pip, etc. in prod)
#     - Faster container startup
# ============================================================

# ── Stage 1: Dependency Builder ─────────────────────────────
FROM python:3.12-slim AS builder

# Set working directory inside the container
WORKDIR /build

# Install system dependencies needed to compile some Python packages
# (psycopg2, cryptography, etc. need C headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST — Docker caches this layer.
# If requirements.txt hasn't changed, this layer is reused
# even if the application code changed. Saves minutes on rebuilds.
COPY requirements.txt .

# Install all Python packages into a specific directory
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime Image ───────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: don't run as root
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

WORKDIR /app

# Install only runtime system libraries (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Switch to non-root user
USER appuser

# Document which port the app listens on
EXPOSE 8000

# Health check — Docker will restart the container if this fails
# Checks every 30s, fails after 3 retries
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Run the application ──────────────────────────────────────
# --host 0.0.0.0   → listen on all interfaces (not just localhost)
# --workers 4      → 4 parallel worker processes (CPU-bound)
# --loop uvloop    → faster async event loop
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop"]
