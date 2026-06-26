# ── Base image ────────────────────────────────────────────────────────────────
# python:3.10-slim is a minimal Debian image with Python pre-installed.
# "slim" removes documentation and testing files — much smaller than the full image.
FROM python:3.10-slim

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────────────────────────────────
# gcc is needed to compile some Python packages from source
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer.
# If your code changes but requirements stay the same, pip install is skipped.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/            ./src/
COPY configs/        ./configs/

# ── Copy model files and metadata ─────────────────────────────────────────────
COPY models/saved/   ./models/saved/
COPY data/processed/feature_metadata.json ./data/processed/

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# --host 0.0.0.0 makes the server reachable from outside the container
# --workers 1 keeps memory use manageable (models are large)
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]