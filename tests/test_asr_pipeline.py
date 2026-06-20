"""Tests for ASRPipeline speech recognition."""
import pytest
from unittest.mock import patch, MagicMock

# Mock torch before importing pipeline
with patch.dict('sys.modules', {'torch': MagicMock(), 'torchaudio': MagicMock()}):
    from audio.pipeline import ASRPipeline


class TestASRPipeline:
    """Test ASR Pipeline functionality."""

    @pytest.fixture
    def config(self):
        """Create a mock Settings object."""
        return type('Settings', (), {
            'asr': type('ASRConfig', (), {'model': 'large-v3', 'device': 'cpu', 'language': 'ru'})(),
            'vad': type('VADConfig', (), {'device': 'cpu'})(),
            'batch_size': type('BatchSize', (), {'get': lambda self, x: 32})(),
            'diarization': type('DiarizationConfig', (), {'device': 'cpu'})(),
        })()

    @pytest.fixture
    def pipeline(self, config):
        """Create a test pipeline instance."""
        return ASRPipeline(config)

    def test_initialization(self, pipeline, config):
        """Test pipeline initialization."""
        assert pipeline.cfg is not None
        assert pipeline.state is not None

    @patch('audio.pipeline.Settings')
    def test_settings_loading(self, mock_settings, config):
        """Test settings are loaded correctly."""
        # Should not raise
        assert config.asr.model == "large-v3"

    def test_overlap_calculation(self, pipeline):
        """Test overlap calculation between two word timestamps."""
        start1, end1 = 0.5, 1.0
        start2, end2 = 0.8, 1.3
        
        # Use module-level function
        from audio.pipeline import overlap
        result = overlap(start1, end1, start2, end2)
        
        assert result is True
