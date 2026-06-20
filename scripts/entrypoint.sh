#!/bin/bash
# =============================================================================
# Speaker-ID ASR Pipeline — Container Entrypoint
# Runs warmup, then starts the main application.
# Handles graceful shutdown and process rotation.
# =============================================================================
set -euo pipefail

LOG_PREFIX="[ENTRYPOINT]"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $*"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX ERROR: $*" >&2
}

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WARMUP_SCRIPT="/app/scripts/warmup.py"
HEALTHCHECK_SCRIPT="/app/scripts/healthcheck.sh"
MAIN_MODULE="orchestrator.main"
CALL_COUNTER_FILE="/tmp/call_counter"
MAX_CALLS_BEFORE_RESTART="${MAX_CALLS_BEFORE_RESTART:-20}"

# -----------------------------------------------------------------------------
# Signal Handlers
# -----------------------------------------------------------------------------
shutdown() {
    log "Received shutdown signal, stopping gracefully..."
    if [[ -n "${APP_PID:-}" ]]; then
        kill -TERM "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    log "Shutdown complete"
    exit 0
}

trap shutdown SIGTERM SIGINT

# -----------------------------------------------------------------------------
# Call Counter (for process rotation)
# -----------------------------------------------------------------------------
init_call_counter() {
    echo 0 > "$CALL_COUNTER_FILE"
}

increment_call_counter() {
    local count
    count=$(cat "$CALL_COUNTER_FILE" 2>/dev/null || echo 0)
    count=$((count + 1))
    echo "$count" > "$CALL_COUNTER_FILE"
    echo "$count"
}

should_rotate() {
    local count
    count=$(cat "$CALL_COUNTER_FILE" 2>/dev/null || echo 0)
    [[ $count -ge $MAX_CALLS_BEFORE_RESTART ]]
}

# -----------------------------------------------------------------------------
# Warmup
# -----------------------------------------------------------------------------
run_warmup() {
    log "Running model warmup..."

    if [[ ! -f "$WARMUP_SCRIPT" ]]; then
        error "Warmup script not found: $WARMUP_SCRIPT"
        return 1
    fi

    # Run warmup with timeout (5 minutes max)
    if timeout 300 python3 "$WARMUP_SCRIPT"; then
        touch /tmp/warmup_complete
        log "Warmup completed successfully"
        return 0
    else
        local exit_code=$?
        error "Warmup failed (exit code: $exit_code)"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
start_app() {
    log "Starting main application: python -m $MAIN_MODULE"
    python3 -m "$MAIN_MODULE" &
    APP_PID=$!
    log "Application started with PID $APP_PID"
}

# -----------------------------------------------------------------------------
# Monitor Loop (handles rotation)
# -----------------------------------------------------------------------------
monitor_loop() {
    while true; do
        if ! kill -0 "$APP_PID" 2>/dev/null; then
            log "Application process $APP_PID exited"
            wait "$APP_PID"
            local exit_code=$?
            log "Application exited with code $exit_code"
            return $exit_code
        fi

        # Check for rotation
        if should_rotate; then
            log "Max calls ($MAX_CALLS_BEFORE_RESTART) reached, initiating rotation..."
            kill -TERM "$APP_PID" 2>/dev/null || true
            wait "$APP_PID" 2>/dev/null || true
            log "Rotation complete, container will restart via Docker policy"
            return 0  # Clean exit triggers docker restart
        fi

        sleep 5
    done
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    log "=" * 60
    log "🚀 Speaker-ID ASR Pipeline Starting"
    log "=" * 60

    # Initialize call counter
    init_call_counter

    # Run warmup (blocking)
    if ! run_warmup; then
        error "Warmup failed, exiting"
        exit 1
    fi

    # Start application
    start_app

    # Monitor
    monitor_loop
    exit $?
}

main "$@"