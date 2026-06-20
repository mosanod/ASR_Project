# Phase 0: Infrastructure — Speaker-ID ASR Pipeline

> **Статус**: Production-Validated  
> **Цель**: Надёжное, наблюдаемое, самоочищающееся окружение для 24/7 обработки звонков.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Host (WSL2/Ubuntu)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   speaker-id │  │    qdrant    │  │    redis     │          │
│  │     app      │◄─┤  (vectors)   │  │  (locks/cache)       │
│  │  (pipeline)  │  │  persistent  │  │              │          │
│  └──────┬───────┘  └──────────────┘  └──────────────┘          │
│         │                                                        │
│  ┌──────▼──────────────────────────────────────────────────┐   │
│  │  Named Volumes (persist across container recreation)    │   │
│  │  • models_volume      — HF/NeMo weights cache           │   │
│  │  • data_volume        — input/output audio, JSON, TXT   │   │
│  │  • known_voices_volume— enrollment references           │   │
│  │  • qdrant_storage_volume — vector DB persistence        │   │
│  │  • redis_volume       — Redis persistence               │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ключевые решения

### 1. VRAM Management — Process Rotation, не `gc.collect()`

**Проблема**: PyTorch Caching Allocator фрагментирует память. После 10-20 звонков `torch.cuda.empty_cache()` показывает свободные ГБ, но аллокация падает с OOM.

**Решение**: 
- Счётчик звонков в `/tmp/call_counter`
- При достижении `MAX_CALLS_BEFORE_RESTART=20` — graceful shutdown → Docker `restart: on-failure` поднимает свежий контейнер
- Warmup запускается при каждом старте (см. ниже)

### 2. Warmup — Обязательный этап Healthcheck

Контейнер **не становится healthy** пока не завершится `warmup.py`:
- Загружает все модели в VRAM
- Прогоняет dummy inference (1 сек тишины)
- Создаёт `/tmp/warmup_complete` флаг
- Healthcheck ждёт до 3 минут (`start_period: 180s`)

Это исключает "первый запрос = 60 сек" в продакшене.

### 3. Qdrant — Persistent, не In-Memory

- `qdrant_storage_volume` монтируется в `/qdrant/storage`
- Две коллекции: `speakers_prod` (основная) + `speakers_staging` (Unknown кластеры, TTL 24h)
- Payload indexes на `phone`, `speaker_id`, `is_admin`, `vector_type` — мгновенный поиск

### 4. Config — Single Source of Truth

- `config/config.yaml` — все параметры
- `config/settings.py` — Pydantic Settings с валидацией на старте
- Environment variables (`SPEAKER_ID__*`) перекрывают YAML
- Ошибка конфига = падение на старте, а не в середине работы

### 5. Non-root User

- Dockerfile создаёт `appuser` (UID 1000)
- Все файлы `chown appuser:appuser`
- Запуск от непривилегированного пользователя

---

## Быстрый старт

```bash
# 1. Клонировать проект
git clone <repo> speaker-id-pipeline
cd speaker-id-pipeline

# 2. Подготовить .env
cp .env.example .env
# Отредактировать при необходимости

# 3. Положить LLM модель (GGUF) в models/llm/
mkdir -p models/llm
# wget .../model.gguf -O models/llm/model.gguf

# 4. Положить эталоны голосов в known_voices/
mkdir -p known_voices
# cp /path/to/Иван_79001234567.wav known_voices/
# cp /path/to/ADMIN_МойГолос.wav known_voices/

# 5. Запуск
docker compose -f docker/docker-compose.yml up -d --build

# 6. Проверить статус
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f app
```

---

## Ожидаемые логи при успешном старте

```
[HEALTHCHECK] Starting healthcheck...
[HEALTHCHECK] GPU OK: 14321MB VRAM free
[HEALTHCHECK] Python imports OK
[HEALTHCHECK] Qdrant OK: collections=[speakers_prod speakers_staging]
[HEALTHCHECK] Redis OK
[HEALTHCHECK] Disk OK: /app/models has 45GB free
[HEALTHCHECK] Models cache OK: ~127 weight files
[HEALTHCHECK] ✅ All checks passed

[ENTRYPOINT] 🚀 Speaker-ID ASR Pipeline Starting
[ENTRYPOINT] Running model warmup...
[WARMUP] 🔥 Starting Model Warmup
[WARMUP] VRAM initial: allocated=0.12GB reserved=0.45GB free=15.23GB
[WARMUP] Warming up Faster-Whisper...
[WARMUP] VRAM after Whisper: allocated=2.34GB reserved=3.12GB free=12.56GB
[WARMUP] Warming up torchaudio CTC alignment...
[WARMUP] VRAM after Alignment: allocated=2.34GB reserved=3.12GB free=12.56GB
[WARMUP] Warming up Silero VAD...
[WARMUP] VRAM after Silero VAD: allocated=2.41GB reserved=3.20GB free=12.48GB
[WARMUP] Warming up NeMo TitaNet...
[WARMUP] VRAM after NeMo TitaNet: allocated=3.87GB reserved=4.92GB free=10.76GB
[WARMUP] Warming up llama-cpp-python...
[WARMUP] VRAM after llama-cpp: allocated=5.12GB reserved=6.45GB free=9.23GB
[WARMUP] Qdrant connectivity OK
[WARMUP] ✅ Warmup completed in 42.3s
[ENTRYPOINT] Warmup completed successfully
[ENTRYPOINT] Starting main application: python -m orchestrator.main
```

---

## Мониторинг

### Prometheus Metrics (порт 9090, путь `/metrics`)

| Метрика | Тип | Описание |
|---------|-----|----------|
| `speaker_id_calls_total` | Counter | Всего обработано звонков |
| `speaker_id_calls_failed_total` | Counter | Звонков с ошибкой |
| `speaker_id_stage_latency_ms` | Histogram | Латенси по этапам (asr, diarization, merger, llm, enrich) |
| `speaker_id_vram_peak_mb` | Gauge | Пиковое VRAM за звонок |
| `speaker_id_llm_circuit_open` | Gauge | 1 если circuit breaker открыт |
| `speaker_id_call_counter` | Gauge | Счётчик звонков до ротации |

### Grafana Dashboard

Импортируйте `monitoring/grafana_dashboard.json` (создаётся на следующем этапе).

---

## Процесс ротации (VRAM Protection)

```
Call 1 ──────► Call 2 ──────► ... ──────► Call 20 ──────► Rotation
    │                                            │
    ▼                                            ▼
warmup_done                            kill -TERM → wait → exit(0)
    │                                            │
    ▼                                            ▼
Docker restart policy (on-failure) → fresh container → warmup → healthy
```

- `MAX_CALLS_BEFORE_RESTART=20` (настраиваемый)
- При VRAM < 1.5 ГБ — WARNING в логах, но ротация по счётчику гарантирована
- Никаких `gc.collect()` в коде пайплайна — только процесс-ротация

---

## Troubleshooting

| Симптом | Причина | Решение |
|---------|---------|---------|
| Container stuck in `starting` | Warmup > 3 min | Увеличьте `start_period` в docker-compose, проверьте VRAM |
| `CUDA out of memory` на звонке 15 | Фрагментация | Уменьшите `MAX_CALLS_BEFORE_RESTART` до 10-15 |
| Qdrant `connection refused` | Qdrant не healthy | Проверьте `docker logs qdrant`, дисковое пространство |
| LLM не отвечает | Circuit breaker открыт | Проверьте `speaker_id_llm_circuit_open=1`, модель GGUF существует |
| `config validation error` | Неверный YAML | `docker run --rm speaker-id-app python -m config.settings` |

---

## Обновление моделей

```bash
# Остановить
docker compose -f docker/docker-compose.yml down

# Удалить старый кэш (опционально)
docker volume rm speaker-id-pipeline_models_volume

# Пересобрать с новым базовым образом
docker compose -f docker/docker-compose.yml up -d --build --pull always
```

---

## Безопасность

- Контейнер запускает от `appuser` (non-root)
- Нет exposed портов кроме внутренних (qdrant 6333, redis 6379, metrics 9090)
- GPU доступ через `nvidia` runtime (только compute/utility capabilities)
- Secrets через `.env` (не коммитить!)

---

## Следующие этапы

- **Phase 1**: Audio Pipeline & Enrollment (`audio/`, `enrollment/`, `qdrant/schema.py`)
- **Phase 2**: ASR & Alignment (`asr/`)
- **Phase 3**: Diarization (`diarization/`)
- **Phase 4**: Merger (`merger/`)
- **Phase 5**: Post-Processing (`postprocess/`)
- **Phase 5.5**: Learning Loop (`learning/`)
- **Phase 6**: Orchestrator (`orchestrator/`)

Каждый этап — отдельный PR с тестами и обновлённой документацией.