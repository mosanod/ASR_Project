#!/usr/bin/env python3
# =============================================================================
# Speaker-ID ASR Pipeline — Model Warmup Script
# Runs at container startup (healthcheck dependency) to load all models into VRAM
# and run a dummy inference. Prevents first-request latency spikes.
# =============================================================================
import sys
import time
import logging
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, "/app")

import torch
import torchaudio
from loguru import logger

from config.settings import get_settings

settings = get_settings()


def setup_logging():
    """Configure loguru for warmup."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.app.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>WARMUP</cyan> | {message}",
        serialize=settings.app.log_format == "json",
    )


def log_vram(tag: str = ""):
    """Log current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
        logger.info(f"VRAM {tag}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB")
    else:
        logger.warning("CUDA not available")


def create_dummy_audio(duration_sec: float = 1.0, sample_rate: int = 16000) -> torch.Tensor:
    """Create silent audio tensor for dummy inference."""
    return torch.zeros(1, int(sample_rate * duration_rate), dtype=torch.float32)


def warmup_faster_whisper():
    """Load Faster-Whisper model and run dummy transcription."""
    logger.info("Warming up Faster-Whisper...")
    from faster_whisper import WhisperModel

    model = WhisperModel(
        settings.asr.model,
        device=settings.asr.device,
        compute_type=settings.asr.compute_type,
        download_root=str(settings.paths.models_dir / "faster-whisper"),
    )

    # Dummy audio: 1 second of silence
    dummy_audio = torch.zeros(16000, dtype=torch.float32).numpy()

    segments, info = model.transcribe(
        dummy_audio,
        language=settings.asr.language,
        beam_size=1,
        vad_filter=False,
        without_timestamps=True,
    )
    list(segments)  # Consume generator

    log_vram("after Whisper")
    logger.info(f"Whisper warmup done (language={info.language}, probability={info.language_probability:.2f})")
    return model


def warmup_whisperx_alignment():
    """Load WhisperX alignment model (Wav2Vec2) - OPTIONAL, only if used."""
    # NOTE: We use torchaudio CTC alignment instead (CPU), so this is skipped.
    # Kept for reference if WhisperX is needed later.
    logger.info("Skipping WhisperX (using torchaudio CTC alignment instead)")
    return None


def warmup_torchaudio_alignment():
    """Load torchaudio CTC forced alignment pipeline."""
    logger.info("Warming up torchaudio CTC alignment...")
    import torchaudio.pipelines as pipelines

    bundle = getattr(pipelines, settings.alignment.model.upper().replace("-", "_"))
    model = bundle.get_model().to(settings.alignment.device)
    model.eval()

    # Dummy inference
    dummy_waveform = torch.randn(1, 16000).to(settings.alignment.device)
    with torch.inference_mode():
        _ = model(dummy_waveform)

    log_vram("after Alignment")
    logger.info("torchaudio alignment warmup done")
    return model


def warmup_silero_vad():
    """Load Silero VAD model."""
    logger.info("Warming up Silero VAD...")
    from silero_vad import load_silero_vad

    model = load_silero_vad(onnx=False)
    model.to(settings.vad.device)

    # Dummy inference
    dummy = torch.randn(1, 16000).to(settings.vad.device)
    with torch.inference_mode():
        _ = model(dummy)

    log_vram("after Silero VAD")
    logger.info("Silero VAD warmup done")
    return model


def warmup_nemo_titanet():
    """Load NeMo TitaNet speaker embedding model."""
    logger.info("Warming up NeMo TitaNet...")
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name=settings.diarization.embedding_model,
        map_location=settings.diarization.device,
    )
    model.eval()

    # Dummy inference: batch of 2 segments
    dummy_batch = torch.randn(2, 1, 16000).to(settings.diarization.device)
    with torch.inference_mode():
        _ = model.forward(input_signal=dummy_batch, input_signal_length=torch.tensor([16000, 16000]))

    log_vram("after NeMo TitaNet")
    logger.info("NeMo TitaNet warmup done")
    return model


def warmup_llama_cpp():
    """Load llama-cpp-python model (if enabled)."""
    if not settings.llm.enabled:
        logger.info("LLM disabled, skipping llama-cpp warmup")
        return None

    logger.info("Warming up llama-cpp-python...")
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(settings.llm.model_path),
        n_ctx=settings.llm.n_ctx,
        n_gpu_layers=settings.llm.n_gpu_layers,
        verbose=False,
    )

    # Dummy inference
    _ = llm("Тест", max_tokens=1, temperature=0.0)

    log_vram("after llama-cpp")
    logger.info("llama-cpp warmup done")
    return llm


def warmup_qdrant():
    """Test Qdrant connectivity."""
    logger.info("Testing Qdrant connection...")
    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        grpc_port=settings.qdrant.grpc_port,
        prefer_grpc=settings.qdrant.prefer_grpc,
        timeout=10,
    )

    # Check collections exist
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    logger.info(f"Qdrant collections: {names}")

    required = [settings.qdrant.collections.production, settings.qdrant.collections.staging]
    for req in required:
        if req not in names:
            logger.warning(f"Collection '{req}' not found — will be created on first use")

    logger.info("Qdrant connectivity OK")
    return client


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("🔥 Starting Model Warmup")
    logger.info("=" * 60)

    start = time.time()
    log_vram("initial")

    try:
        # Load models in order of VRAM usage (heaviest first)
        # Keep references to prevent GC during warmup
        models = {}

        models["whisper"] = warmup_faster_whisper()
        models["alignment"] = warmup_torchaudio_alignment()
        models["vad"] = warmup_silero_vad()
        models["titanet"] = warmup_nemo_titanet()
        models["llm"] = warmup_llama_cpp()
        models["qdrant"] = warmup_qdrant()

        elapsed = time.time() - start
        log_vram("final")
        logger.success(f"✅ Warmup completed in {elapsed:.1f}s")
        return 0

    except Exception as e:
        logger.exception(f"❌ Warmup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())