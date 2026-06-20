# =============================================================================
# Speaker-ID ASR Pipeline — Production Docker Image
# Base: PyTorch 2.4 + CUDA 12.4 + cuDNN 9 (runtime)
# =============================================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Non-root user (security best practice)
# -----------------------------------------------------------------------------
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

# -----------------------------------------------------------------------------
# Python dependencies (cached layer)
# -----------------------------------------------------------------------------
WORKDIR /app

# Copy requirements first for better layer caching
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Application code
# -----------------------------------------------------------------------------
COPY --chown=appuser:appuser . /app

# -----------------------------------------------------------------------------
# Volumes (mount points for persistence)
# -----------------------------------------------------------------------------
# /app/models       — HF/NeMo model cache (avoid re-download on rebuild)
# /app/data         — input audio + output JSON/TXT
# /app/known_voices — enrollment references (NAME_PHONE.wav)
# /app/qdrant_storage — Qdrant persistent storage (NOT in-memory for prod)
VOLUME ["/app/models", "/app/data", "/app/known_voices", "/app/qdrant_storage"]

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models/huggingface \
    TORCH_HOME=/app/models/torch \
    NEMO_CACHE_DIR=/app/models/nemo \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -----------------------------------------------------------------------------
# Healthcheck (validates GPU, models, Qdrant connectivity)
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD /app/scripts/healthcheck.sh

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
USER appuser
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["python", "-m", "orchestrator.main"]