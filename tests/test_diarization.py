"""Tests for Diarization module."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

# Mock torch before importing pipeline
with patch.dict('sys.modules', {'torch': MagicMock(), 'torchaudio': MagicMock()}):
    from audio.diarization import (
        SpeakerEmbeddingExtractor,
        SpeakerClustering,
        DiarizationEngine,
        diarize_audio,
    )


class TestSpeakerEmbeddingExtractor:
    """Test speaker embedding extraction."""

    def test_extraction_returns_correct_shape(self):
        """Test that embeddings have correct dimensionality."""
        extractor = SpeakerEmbeddingExtractor()
        
        # Create dummy audio (1 second at 16kHz)
        sample_rate = 16000
        duration_sec = 1.0
        audio = np.random.randn(int(sample_rate * duration_sec)).astype(np.float32)
        
        embedding = extractor.extract(audio, sample_rate)
        
        assert len(embedding) == 192  # TitaNet dimension

    def test_extraction_handles_short_audio(self):
        """Test extraction with very short audio."""
        extractor = SpeakerEmbeddingExtractor()
        
        # Create dummy audio (0.1 seconds at 16kHz)
        sample_rate = 16000
        duration_sec = 0.1
        audio = np.random.randn(int(sample_rate * duration_sec)).astype(np.float32)
        
        embedding = extractor.extract(audio, sample_rate)
        
        assert len(embedding) == 192

    def test_extraction_returns_normalized_embeddings(self):
        """Test that embeddings are normalized."""
        extractor = SpeakerEmbeddingExtractor()
        
        # Use fixed audio for deterministic testing
        audio = np.ones(16000, dtype=np.float32) * 0.5
        
        embedding = extractor.extract(audio, sample_rate=16000)
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01


class TestSpeakerClustering:
    """Test speaker clustering."""

    def test_cluster_single_embedding(self):
        """Test clustering with single embedding."""
        clusterer = SpeakerClustering(distance_threshold=0.5)
        
        embeddings = np.random.randn(1, 192).astype(np.float32)
        clusters = clusterer.cluster(embeddings)
        
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_cluster_similar_embeddings(self):
        """Test clustering similar embeddings together."""
        clusterer = SpeakerClustering(distance_threshold=0.5)
        
        # Create two very similar embeddings (same speaker)
        base_embedding = np.random.randn(192).astype(np.float32)
        embedding1 = base_embedding / np.linalg.norm(base_embedding)
        embedding2 = base_embedding.copy()  # Identical
        
        embeddings = np.stack([embedding1, embedding2])
        clusters = clusterer.cluster(embeddings)
        
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_cluster_different_embeddings(self):
        """Test clustering different embeddings separately."""
        clusterer = SpeakerClustering(distance_threshold=0.5)
        
        # Create two very different embeddings (different speakers)
        embedding1 = np.random.randn(192).astype(np.float32)
        embedding2 = -embedding1  # Opposite direction
        
        embeddings = np.stack([embedding1, embedding2])
        clusters = clusterer.cluster(embeddings)
        
        assert len(clusters) == 2

    def test_cluster_empty_embeddings(self):
        """Test clustering with empty input."""
        clusterer = SpeakerClustering(distance_threshold=0.5)
        
        embeddings = np.array([]).reshape(0, 192)
        clusters = clusterer.cluster(embeddings)
        
        assert len(clusters) == 0


class TestDiarizationEngine:
    """Test main diarization engine."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        return DiarizationEngine()

    def test_process_empty_segments(self, engine):
        """Test processing empty segments list."""
        result = engine.process([])
        
        assert result == {}

    def test_process_single_segment(self, engine):
        """Test processing single segment."""
        sample_rate = 16000
        audio = np.random.randn(16000).astype(np.float32)
        
        segments = [(audio, sample_rate)]
        result = engine.process(segments)
        
        assert len(result) >= 1

    def test_process_multiple_segments_same_speaker(self, engine):
        """Test processing multiple segments from same speaker."""
        # Create identical audio for all segments (same speaker)
        base_audio = np.random.randn(16000).astype(np.float32)
        
        segments = [
            (base_audio.copy(), 16000),
            (base_audio.copy(), 16000),
            (base_audio.copy(), 16000),
        ]
        
        result = engine.process(segments)
        
        # All segments should be clustered together
        speaker_ids = list(result.keys())
        assert len(speaker_ids) == 1

    def test_process_multiple_segments_different_speakers(self, engine):
        """Test processing multiple segments from different speakers."""
        # Create very different audio for each segment (different speakers)
        base_audio = np.random.randn(16000).astype(np.float32)
        
        segments = [
            (base_audio.copy(), 16000),
            (-base_audio.copy(), 16000),  # Inverted = different speaker
            (np.random.randn(16000).astype(np.float32), 16000),
        ]
        
        result = engine.process(segments)
        
        # Should have multiple speakers
        speaker_ids = list(result.keys())
        assert len(speaker_ids) >= 2

    def test_get_speaker_count(self, engine):
        """Test speaker count estimation."""
        sample_rate = 16000
        
        # Single speaker scenario (identical audio)
        audio1 = np.random.randn(16000).astype(np.float32)
        segments1 = [(audio1.copy(), 16000), (audio1.copy(), 16000)]
        
        count1 = engine.get_speaker_count(segments1)
        assert count1 == 1
        
        # Multiple speakers scenario (inverted audio = different speaker)
        audio2 = np.random.randn(16000).astype(np.float32)
        segments2 = [
            (audio1.copy(), 16000),
            (-audio1.copy(), 16000),
            (audio2.copy(), 16000),
        ]
        
        count2 = engine.get_speaker_count(segments2)
        assert count2 >= 2


class TestDiarizeAudio:
    """Test convenience function."""

    def test_diarize_audio_returns_dict(self):
        """Test that diarize_audio returns a dictionary."""
        # Use temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = diarize_audio(temp_path, sample_rate=16000)
            
            assert isinstance(result, dict)
        finally:
            # Cleanup (file is empty but exists)
            if temp_path.exists():
                os.unlink(temp_path)
