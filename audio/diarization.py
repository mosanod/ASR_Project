"""
Speaker Diarization Module — Independent Implementation
========================================================
Uses speaker embeddings (TitaNet) + cosine similarity clustering.
No external diarization dependencies required.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings from audio segments."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        # Placeholder for actual embedding model loading
        self._model_loaded = False

    def _load_model(self):
        """Load TitaNet or ECAPA model (placeholder)."""
        if not self._model_loaded:
            print(f"Loading speaker embedding model on {self.device}...")
            # TODO: Load actual model from HuggingFace
            # from nemo.collections.asr.models import SpeakerClassificationModels
            # self.model = SpeakerClassificationModels.titanet_large
            self._model_loaded = True

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract speaker embedding from audio segment."""
        self._load_model()
        
        # Placeholder: return deterministic embeddings for testing
        # In production: use actual model inference
        embedding_dim = 192  # TitaNet dimension
        
        # Use hash of audio content to generate deterministic "embeddings"
        audio_hash = int.from_bytes(audio.tobytes()[:32], 'little')
        np.random.seed(hash(audio_hash) % (2**31))
        
        n_frames = min(30, len(audio) // sample_rate)  # Max 30 seconds
        
        if n_frames <= 0:
            return np.zeros(embedding_dim)
        
        # Generate synthetic embeddings (replace with actual inference)
        embedding = np.random.randn(n_frames, embedding_dim).astype(np.float32)
        
        # Average pooling over time + normalization
        mean_emb = embedding.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            return mean_emb / norm
        
        return mean_emb


class SpeakerClustering:
    """Cluster speaker embeddings using hierarchical clustering."""

    def __init__(self, distance_threshold: float = 0.5, linkage: str = "average"):
        self.distance_threshold = distance_threshold
        self.linkage = linkage

    def cluster(self, embeddings: np.ndarray) -> List[np.ndarray]:
        """Cluster embeddings into speaker groups."""
        if len(embeddings) == 0:
            return []
        
        # Normalize embeddings to unit sphere
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms
        
        # Simple k-means-like clustering (replace with proper hierarchical)
        if len(normalized) <= self.distance_threshold:
            return [normalized.copy()]
        
        # Greedy clustering
        clusters = []
        used = np.zeros(len(normalized), dtype=bool)
        
        for i in range(len(normalized)):
            if used[i]:
                continue
            
            cluster = [i]
            used[i] = True
            
            for j in range(i + 1, len(normalized)):
                if used[j]:
                    continue
                
                # Cosine similarity
                sim = np.dot(normalized[i], normalized[j])
                
                if sim > self.distance_threshold:
                    cluster.append(j)
                    used[j] = True
            
            clusters.append(np.array([normalized[k] for k in cluster]))
        
        return clusters

    def get_speaker_ids(self, embeddings: List[np.ndarray]) -> Dict[int, int]:
        """Assign speaker IDs to each embedding."""
        ids = {}
        current_id = 0
        
        for i, emb in enumerate(embeddings):
            if i not in ids:
                ids[i] = current_id
                current_id += 1
        
        return ids


class DiarizationEngine:
    """Main diarization engine combining extraction and clustering."""

    def __init__(self, device: str = "cpu", distance_threshold: float = 0.5):
        self.device = device
        self.embedding_extractor = SpeakerEmbeddingExtractor(device)
        self.clusterer = SpeakerClustering(distance_threshold=distance_threshold)

    def process(self, audio_segments: List[Tuple[np.ndarray, int]]) -> Dict[int, List[Dict]]:
        """Process all segments and assign speaker IDs."""
        if not audio_segments:
            return {}
        
        # Extract embeddings for each segment
        all_embeddings = []
        segment_info = []
        
        for idx, (audio, sample_rate) in enumerate(audio_segments):
            embedding = self.embedding_extractor.extract(audio, sample_rate)
            all_embeddings.append(embedding)
            
            segment_info.append({
                "segment_idx": idx,
                "duration_sec": len(audio) / sample_rate if sample_rate > 0 else 0,
            })
        
        # Cluster embeddings
        clusters = self.clusterer.cluster(np.array(all_embeddings))
        
        # Assign speaker IDs
        speaker_ids = {}
        for cluster_idx, cluster in enumerate(clusters):
            for local_idx, _ in enumerate(cluster):
                global_idx = sum(len(c) for c in clusters[:cluster_idx]) + local_idx
                speaker_ids[global_idx] = cluster_idx
        
        # Build result
        result = {speaker_id: [] for speaker_id in range(len(clusters))}
        
        for idx, info in enumerate(segment_info):
            if idx in speaker_ids:
                speaker_id = speaker_ids[idx]
                result[speaker_id].append({
                    "segment_idx": info["segment_idx"],
                    "duration_sec": info["duration_sec"],
                })
        
        return result

    def get_speaker_count(self, audio_segments: List[Tuple[np.ndarray, int]]) -> int:
        """Estimate number of speakers in the recording."""
        if not audio_segments:
            return 0
        
        embeddings = [self.embedding_extractor.extract(_audio, sr) for _audio, sr in audio_segments]
        clusters = self.clusterer.cluster(np.array(embeddings))
        
        # Filter small clusters (noise)
        valid_clusters = [c for c in clusters if len(c) >= 1]
        return len(valid_clusters)


def diarize_audio(audio_path: Path, sample_rate: int = 16000) -> Dict[int, List[Dict]]:
    """Convenience function to diarize an audio file."""
    # Load audio (placeholder)
    audio_data = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds
    
    engine = DiarizationEngine()
    segments = [(audio_data, sample_rate)]
    
    return engine.process(segments)
