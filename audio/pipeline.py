"""
Speaker-ID ASR Pipeline - Core Implementation
==============================================
Orchestrates speech recognition (ASR) and speaker diarization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from config.settings import Settings, get_settings


@dataclass
class Segment:
    """Single transcription segment with speaker info."""
    start: float
    end: float
    text: str
    speaker_id: Optional[str] = None
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Result of ASR transcription with diarization."""
    segments: List[Segment]
    audio_info: Dict[str, Any] = field(default_factory=dict)
    total_duration: float = 0.0


class PipelineState:
    """Runtime state for the pipeline."""

    def __init__(self):
        self.asr_model = None
        self.diarizer = None
        self.vad_pipeline = None


def overlap(start1: float, end1: float, start2: float, end2: float) -> bool:
    """Check if two time intervals overlap."""
    return not (end1 <= start2 or end2 <= start1)


class ASRPipeline:
    """
    End-to-end ASR and Diarization Pipeline.

    Coordinates Whisper-based transcription with speaker diarization,
    including voice activity detection and speaker embedding extraction.
    """

    def __init__(self, config: Settings):
        self.cfg = config
        self.state = PipelineState()
        self._asr_model = None
        self.vad_model = None

    @property
    def asr_model(self):
        """Initialize Whisper ASR model."""
        if self._asr_model is None:
            from faster_whisper import WhisperModel
            model_size = getattr(self.cfg.asr, 'model_size', 'large-v3')
            device = torch.device(self.cfg.asr.device)
            self._asr_model = WhisperModel(
                model_size_or_path=model_size,
                device=str(device),
                compute_type="float16",
                download_root=self.cfg.asr.download_dir or "./models/whisper"
            )
        return self._asr_model

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file."""
        import librosa
        waveform, sample_rate = librosa.load(
            str(audio_path),
            sr=16000,
            mono=True,
            dtype=np.float32
        )
        return waveform, sample_rate

    def _detect_voice_activity(self, waveform: np.ndarray, sample_rate: int) -> List[Dict[str, float]]:
        """Detect speech activity using Silero VAD."""
        import torch
        from silero_vad import get_speech_timestamps, load_model

        # Load or get cached model
        if self.vad_model is None:
            self.vad_model, _ = load_model()

        vad_params = getattr(self.cfg, 'params', {}) or {}
        vad_params.setdefault("threshold", 0.5)
        vad_params.setdefault("min_speech_duration_ms", 250)
        vad_params.setdefault("min_silence_duration_ms", 100)

        waveform_tensor = torch.from_numpy(waveform).float()

        try:
            timestamps = get_speech_timestamps(
                waveform_tensor,
                self.vad_model,
                sampling_rate=sample_rate,
                **vad_params
            )
        except Exception as e:
            raise RuntimeError(f"VAD error: {e}. Check silero_vad API compatibility.")

        segments = []
        for ts in timestamps:
            if isinstance(ts, dict):
                start = ts.get("start", 0) * 1000.0
                end = ts.get("end", 0) * 1000.0
            elif hasattr(ts, 'get):
                start = getattr(ts, "start", 0) * 1000.0
                end = getattr(ts, "end", 0) * 1000.0
            else:
                try:
                    if hasattr(ts, '__len__') and len(ts) >= 2:
                        start = float(ts[0]) * 1000.0
                        end = float(ts[1]) * 1000.0
                    elif hasattr(ts, "end"):
                        start = getattr(ts, "start", 0) * 1000.0
                        end = ts.end * 1000.0
                    else:
                        total_samples = len(waveform)
                        sample_duration = 1.0 / sample_rate
                        end_ms = total_samples * sample_duration * 1000
                        start = 0.0
                        end = end_ms
                except (TypeError, AttributeError):
                    continue

            if not segments or start != segments[-1].get("start"):
                segments.append({"start": start, "end": end})
            else:
                try:
                    if hasattr(ts, '__len__') and len(ts) >= 2:
                        start = float(ts[0]) * 1000.0
                        end = float(ts[1]) * 1000.0
                    elif hasattr(ts, "end"):
                        start = getattr(ts, "start", 0) * 1000.0
                        end = ts.end * 1000.0
                    else:
                        total_samples = len(waveform)
                        sample_duration = 1.0 / sample_rate
                        end_ms = total_samples * sample_duration * 1000
                        start = 0.0
                        end = end_ms
                except (TypeError, AttributeError):
                    continue

            if not segments or start != segments[-1].get("start"):
                segments.append({"start": start, "end": end})

        if not segments:
            return [{"start": 0.0, "end": len(waveform) / sample_rate * 1000}]

        return segments

    def _recognize_speech(self, vad_segments: List[Dict[str, float]],
                          waveform: np.ndarray, sample_rate: int) -> List[Segment]:
        """Transcribe each VAD segment using Whisper."""
        results = []
        for seg in vad_segments:
            start_ms = seg["start"]
            end_ms = seg["end"]

            start_idx = int(start_ms * sample_rate / 1000.0)
            end_idx = int(end_ms * sample_rate / 1000.0)

            segment_audio = waveform[start_idx:end_idx]

            try:
                segments, _ = self.asr_model.transcribe(
                    segment_audio,
                    language='ru',
                    task="transcribe",
                    beam_size=5,
                    condition_on_previous_text=False
                )

                for seg in segments:
                    text = getattr(seg, 'text', '').strip()
                    if not text:
                        continue

                    results.append(Segment(
                        start=start_ms / 1000.0,
                        end=end_ms / 1000.0,
                        text=text,
                        confidence=0.9
                    ))
            except Exception as e:
                print(f"ASR error: {e}")
                continue

        return results

    def process(self, audio_path: Path) -> TranscriptionResult:
        """Process audio file and return transcription result."""
        waveform, sample_rate = self._load_audio(audio_path)

        vad_segments = self._detect_voice_activity(waveform, sample_rate)
        print(f"VAD detected {len(vad_segments)} segments")

        segments = self._recognize_speech(vad_segments, waveform, sample_rate)

        return TranscriptionResult(
            segments=segments,
            total_duration=len(waveform) / sample_rate if len(waveform) > 0 else 0
        )

    def process_batch(self, audio_paths: List[Path]) -> TranscriptionResult:
        """Process multiple audio files."""
        all_segments = []
        total_duration = 0

        for path in audio_paths:
            result = self.process(path)
            all_segments.extend(result.segments)
            if len(audio_paths) == 1:
                return result

        return TranscriptionResult(
            segments=all_segments,
            total_duration=sum(getattr(r, 'total_duration', 0) for r in
                             [self._get_audio_info(p) for p in audio_paths])
        )

    def _get_audio_info(self, path: Path) -> Dict[str, Any]:
        """Get audio file info."""
        import soundfile as sf
        try:
            sig, sample_rate = sf.read(str(path), dtype='float32')
            return {
                'path': path,
                'duration': len(sig) / sample_rate if sample_rate > 0 else 0,
                'channels': 1,
                'sample_rate': sample_rate
            }
        except Exception:
            return {'path': path, 'duration': 0, 'channels': 1, 'sample_rate': 16000}
