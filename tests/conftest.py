"""
Shared pytest fixtures for Speaker-ID ASR Pipeline tests.
========================================================
Provides common test utilities and mock configurations.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# Mock PyTorch before importing any audio modules
@pytest.fixture(autouse=True)
def mock_torch():
    """Ensure torch is mocked for all tests."""
    with patch.dict('sys.modules', {
        'torch': MagicMock(),
        'torchaudio': MagicMock(),
        'pyannote.audio': MagicMock(),
    }):
        yield


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a dummy audio file for testing."""
    import numpy as np
    
    # Create 1 second of silence at 16kHz
    sample_rate = 16000
    duration_sec = 1.0
    samples = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
    
    audio_path = temp_dir / "dummy.wav"
    # Save as WAV (simplified — in production use soundfile or wave)
    import wave
    with wave.open(str(audio_path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for sample in samples:
            wav_file.writeframes((sample * 32767).to_bytes(2, byteorder='little', signed=True))
    
    return audio_path


@pytest.fixture
def mock_settings():
    """Create a mock Settings object for testing."""
    return type('Settings', (), {
        'asr': type('ASRConfig', (), {'model': 'large-v3', 'device': 'cpu', 'language': 'ru'})(),
        'vad': type('VADConfig', (), {'device': 'cpu'})(),
        'batch_size': type('BatchSize', (), {'get': lambda self, x: 32})(),
        'diarization': type('DiarizationConfig', (), {'device': 'cpu'})(),
    })()


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    return MagicMock()
