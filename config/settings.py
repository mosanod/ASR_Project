# =============================================================================
# Speaker-ID ASR Pipeline — Pydantic Settings (Config Validation)
# Single source of truth: config.yaml -> validated Settings object
# =============================================================================
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# Nested Models
# -----------------------------------------------------------------------------
class PathsConfig(BaseSettings):
    models_dir: Path = Path("./models")
    data_dir: Path = Path("./data")
    input_dir: Path = Path("./data/input")
    output_dir: Path = Path("./data/output")
    known_voices_dir: Path = Path("./known_voices")
    qdrant_storage: Path = Path("./qdrant_storage")
    prompts_dir: Path = Path("./prompts")
    processed_calls_log: Path = Path("./data/processed_calls.json")
    processed_enrollments_log: Path = Path("./data/processed_enrollments.json")

    @field_validator("*", mode="before")
    @classmethod
    def expand_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class ASRConfig(BaseSettings):
    model: str = "large-v3"
    language: str = "ru"
    beam_size: int = Field(ge=1, le=10, default=5)
    compute_type: Literal["float16", "int8_float16", "int8"] = "float16"
    device: str = "cpu"  # Changed to cpu for local testing
    initial_prompt_template: str = "Разговор между менеджером и клиентом. Имена: {names}. Термины: {terms}."
    project_terms: List[str] = []
    vad_filter: bool = True
    params: dict = Field(default_factory=lambda: {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 30,
        "min_silence_duration_ms": 100,
    })


class AlignmentConfig(BaseSettings):
    model: str = "wav2vec2_base"
    device: Literal["cpu", "cuda"] = "cpu"
    batch_size: int = Field(ge=1, le=64, default=16)


class VADConfig(BaseSettings):
    model: str = "silero_vad"
    threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    min_speech_duration_ms: int = Field(ge=1, default=500)
    min_silence_duration_ms: int = Field(ge=1, default=200)
    segment_padding_ms: int = Field(ge=0, default=100)
    device: str = "cpu"  # Changed to cpu for local testing
    params: dict = Field(default_factory=lambda: {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 30,
        "min_silence_duration_ms": 100,
    })


class DiarizationConfig(BaseSettings):
    embedding_model: str = "nvidia/ctn-titanet_large"
    batch_size: int = Field(ge=1, le=128, default=32)
    device: str = "cpu"  # Changed to cpu for local testing
    base_similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.75)
    use_dynamic_threshold: bool = True
    dynamic_threshold_multiplier: float = Field(ge=1.0, default=2.5)
    phone_match_bonus: float = Field(ge=0.0, le=0.5, default=0.10)
    min_segment_duration_sec: float = Field(ge=0.1, default=0.5)
    max_segment_duration_sec: float = Field(ge=1.0, default=30.0)


class ClusteringConfig(BaseSettings):
    enable_unknown_clustering: bool = True
    min_unknown_segments: int = Field(ge=1, default=15)
    distance_threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    linkage: Literal["ward", "complete", "average", "single"] = "average"


class MergerConfig(BaseSettings):
    max_pause_ms: int = Field(ge=100, default=1000)
    overlap_tolerance: float = Field(ge=0.0, le=0.5, default=0.15)
    vad_aware_grouping: bool = True


class CircuitBreakerConfig(BaseSettings):
    enabled: bool = True
    failure_threshold: int = Field(ge=1, default=5)
    recovery_timeout_seconds: int = Field(ge=1, default=600)


class LLMConfig(BaseSettings):
    enabled: bool = False  # Disabled for local testing
    model_path: Path = Path("./models/llm/model.gguf")
    n_ctx: int = Field(ge=512, default=8192)
    n_gpu_layers: int = -1
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    top_p: float = Field(ge=0.0, le=1.0, default=0.95)
    repeat_penalty: float = Field(ge=0.0, default=1.1)
    system_prompt_path: Path = Path("./prompts/refine.txt")
    max_tokens: int = Field(ge=1, default=2048)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)

    @field_validator("model_path", "system_prompt_path", mode="before")
    @classmethod
    def expand_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class RulesConfig(BaseSettings):
    enabled: bool = True
    filler_words: List[str] = Field(default_factory=lambda: [
        "ээ", "эээ", "ну", "как бы", "тоже", "короче", "типа", "кстати"
    ])
    uncertain_confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.5)


class LearningConfig(BaseSettings):
    enable_feedback_loop: bool = True
    min_confidence_to_enroll: float = Field(ge=0.0, le=1.0, default=0.85)
    min_segment_duration_sec: float = Field(ge=0.5, default=2.0)
    min_vector_norm: float = Field(ge=0.0, le=1.0, default=0.7)
    update_strategy: Literal["multi_vector"] = "multi_vector"
    max_vectors_per_speaker: int = Field(ge=1, default=50)
    protect_enrolled_vectors: bool = True


class QdrantCollectionConfig(BaseSettings):
    production: str = "speakers_prod"
    staging: str = "speakers_staging"


class QdrantConfig(BaseSettings):
    host: str = "localhost"  # Changed to localhost for local testing
    port: int = Field(ge=1, le=65535, default=6333)
    grpc_port: int = Field(ge=1, le=65535, default=6334)
    prefer_grpc: bool = True
    collections: QdrantCollectionConfig = Field(default_factory=QdrantCollectionConfig)
    vector_size: int = Field(ge=1, default=192)
    distance: Literal["Cosine", "Euclid", "Dot"] = "Cosine"
    hnsw_config: dict = Field(default_factory=lambda: {
        "m": 16,
        "ef_construct": 100,
        "full_scan_threshold": 10000,
    })
    payload_indexes: List[str] = Field(default_factory=lambda: [
        "phone", "speaker_id", "is_admin", "vector_type"
    ])
    staging_ttl_hours: int = Field(ge=1, default=24)


class RedisConfig(BaseSettings):
    host: str = "localhost"  # Changed to localhost for local testing
    port: int = Field(ge=1, le=65535, default=6379)
    db: int = Field(ge=0, le=15, default=0)
    lock_ttl_seconds: int = Field(ge=1, default=3600)
    idempotency_ttl_days: int = Field(ge=1, default=30)


class MonitoringConfig(BaseSettings):
    enabled: bool = True
    port: int = Field(ge=1, le=65535, default=9090)
    path: str = "/metrics"
    vram_log_interval_calls: int = Field(ge=1, default=5)
    vram_warning_threshold_gb: float = Field(ge=0.0, default=1.5)


class AppConfig(BaseSettings):
    name: str = "speaker-id-asr-pipeline"
    version: str = "3.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    max_calls_before_restart: int = Field(ge=1, default=20)
    warmup_on_start: bool = True


# -----------------------------------------------------------------------------
# Root Settings
# -----------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config/config.yaml",
        yaml_file_encoding="utf-8",
        env_prefix="SPEAKER_ID_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    merger: MergerConfig = Field(default_factory=MergerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # -------------------------------------------------------------------------
    # Cross-field validation
    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_consistency(self):
        # Vector size must match embedding model
        if "titanet" in self.diarization.embedding_model.lower():
            expected = 192
        elif "ecapa" in self.diarization.embedding_model.lower():
            expected = 512
        else:
            expected = 192  # Default assumption

        if self.qdrant.vector_size != expected:
            raise ValueError(
                f"qdrant.vector_size ({self.qdrant.vector_size}) must match "
                f"embedding model ({self.diarization.embedding_model}) expected dim ({expected})"
            )

        # Paths must exist or be creatable
        for path_field in [
            "models_dir", "data_dir", "input_dir", "output_dir",
            "known_voices_dir", "qdrant_storage", "prompts_dir"
        ]:
            path = getattr(self.paths, path_field)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValueError(f"Cannot create path {path_field}={path}: {e}")

        # LLM model file existence (only if enabled)
        if self.llm.enabled and not self.llm.model_path.exists():
            # Warning only - model might be downloaded at runtime
            import logging
            logging.getLogger(__name__).warning(
                f"LLM model not found at {self.llm.model_path} — will attempt download or fail at runtime"
            )

        # System prompt existence
        if self.llm.enabled and not self.llm.system_prompt_path.exists():
            raise ValueError(f"LLM system prompt not found: {self.llm.system_prompt_path}")

        return self


# -----------------------------------------------------------------------------
# Singleton instance (import once, use everywhere)
# -----------------------------------------------------------------------------
settings = Settings()


# -----------------------------------------------------------------------------
# Convenience accessors
# -----------------------------------------------------------------------------
def get_settings() -> Settings:
    """Get validated settings instance."""
    return settings


if __name__ == "__main__":
    # Quick validation test
    import json
    print("✅ Config validation passed")
    print(f"App: {settings.app.name} v{settings.app.version}")
    print(f"ASR: {settings.asr.model} ({settings.asr.compute_type})")
    print(f"Diarization: {settings.diarization.embedding_model} -> {settings.qdrant.vector_size}d")
    print(f"Qdrant: {settings.qdrant.host}:{settings.qdrant.port} (prod={settings.qdrant.collections.production})")
    print(f"LLM: {'enabled' if settings.llm.enabled else 'disabled'} -> {settings.llm.model_path}")
