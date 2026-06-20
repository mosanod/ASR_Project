#!/bin/bash
# =============================================================================
# Speaker-ID ASR Pipeline — Healthcheck Script
# Used by Docker HEALTHCHECK and docker-compose healthcheck
# Validates: GPU, Models loaded, Qdrant, Redis, Disk space
# =============================================================================
set -euo pipefail

LOG_PREFIX="[HEALTHCHECK]"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $*"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX ERROR: $*" >&2
}

# -----------------------------------------------------------------------------
# 1. GPU Availability
# -----------------------------------------------------------------------------
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found"
        return 1
    fi

    if ! nvidia-smi -L &> /dev/null; then
        error "No NVIDIA GPUs detected"
        return 1
    fi

    # Check VRAM
    VRAM_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ $VRAM_FREE_MB -lt 1000 ]]; then
        error "Low VRAM: ${VRAM_FREE_MB}MB free (need >1000MB)"
        return 1
    fi

    log "GPU OK: ${VRAM_FREE_MB}MB VRAM free"
    return 0
}

# -----------------------------------------------------------------------------
# 2. Python Environment & Critical Imports
# -----------------------------------------------------------------------------
check_python() {
    python3 -c "
import torch
import torchaudio
import faster_whisper
import nemo.collections.asr as nemo_asr
import qdrant_client
import llama_cpp
import silero_vad
print('Python imports OK')
" 2>&1 | while read line; do log "$line"; done

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        error "Python import check failed"
        return 1
    fi
    return 0
}

# -----------------------------------------------------------------------------
# 3. Qdrant Connectivity
# -----------------------------------------------------------------------------
check_qdrant() {
    local host="${QDRANT_HOST:-qdrant}"
    local port="${QDRANT_PORT:-6333}"

    if ! curl -sf "http://${host}:${port}/health" > /dev/null; then
        error "Qdrant health endpoint unreachable at ${host}:${port}"
        return 1
    fi

    # Check collections exist
    COLLECTIONS=$(curl -sf "http://${host}:${port}/collections" | python3 -c "
import sys, json
data = json.load(sys.stdin)
names = [c['name'] for c in data.get('result', {}).get('collections', [])]
print(' '.join(names))
")

    log "Qdrant OK: collections=[${COLLECTIONS}]"
    return 0
}

# -----------------------------------------------------------------------------
# 4. Redis Connectivity
# -----------------------------------------------------------------------------
check_redis() {
    local host="${REDIS_HOST:-redis}"
    local port="${REDIS_PORT:-6379}"

    if ! redis-cli -h "${host}" -p "${port}" ping | grep -q PONG; then
        error "Redis ping failed at ${host}:${port}"
        return 1
    fi

    log "Redis OK"
    return 0
}

# -----------------------------------------------------------------------------
# 5. Disk Space
# -----------------------------------------------------------------------------
check_disk() {
    local min_free_gb=2

    for dir in "/app/models" "/app/data" "/app/qdrant_storage" "/app/known_voices"; do
        if [[ -d "$dir" ]]; then
            FREE_GB=$(df -BG "$dir" | awk 'NR==2 {print $4}' | sed 's/G//')
            if [[ $FREE_GB -lt $min_free_gb ]]; then
                error "Low disk space on $dir: ${FREE_GB}GB free (need >${min_free_gb}GB)"
                return 1
            fi
            log "Disk OK: $dir has ${FREE_GB}GB free"
        fi
    done
    return 0
}

# -----------------------------------------------------------------------------
# 6. Models Directory (weights cached)
# -----------------------------------------------------------------------------
check_models() {
    local models_dir="/app/models"

    # Check for at least one model cache
    if [[ ! -d "$models_dir" ]]; then
        error "Models directory not found: $models_dir"
        return 1
    fi

    # Count cached models (rough heuristic)
    CACHE_COUNT=$(find "$models_dir" -type f \( -name "*.bin" -o -name "*.pt" -o -name "*.safetensors" -o -name "*.nemo" \) 2>/dev/null | wc -l)

    if [[ $CACHE_COUNT -eq 0 ]]; then
        log "WARNING: No model weights found in $models_dir (cold start expected)"
    else
        log "Models cache OK: ~${CACHE_COUNT} weight files"
    fi
    return 0
}

# -----------------------------------------------------------------------------
# 7. Warmup Completed Flag (optional)
# -----------------------------------------------------------------------------
check_warmup() {
    local flag_file="/tmp/warmup_complete"

    if [[ -f "$flag_file" ]]; then
        log "Warmup flag found"
    else
        log "Warmup flag not found (may still be running)"
    fi
    return 0  # Non-blocking
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log "Starting healthcheck..."

    local failed=0

    check_gpu || failed=1
    check_python || failed=1
    check_qdrant || failed=1
    check_redis || failed=1
    check_disk || failed=1
    check_models || failed=1
    check_warmup || failed=1

    if [[ $failed -eq 0 ]]; then
        log "✅ All checks passed"
        exit 0
    else
        error "❌ One or more checks failed"
        exit 1
    fi
}

main "$@"