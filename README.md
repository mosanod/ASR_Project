# Speaker-ID ASR Pipeline — Руководство по использованию

Система распознавания речи с идентификацией говорящего (Speaker-ID).  
Поддерживает: транскрипцию аудио, диаризацию (разделение по спикерам), кластеризацию голосов.

---

## Что нужно для настоящих тестов

У вас уже есть тестовый файл `test.wav`. Для полноценного запуска нужны:

### 1. Зависимости (установить через pip)
```bash
pip install -r requirements.txt
```

### 2. Модель Faster-Whisper (ASR)
Первый запуск автоматически скачает модель. По умолчанию используется `large-v3`.  
Можно сменить в `config/config.yaml` или `.env`:
```yaml
asr:
  model: "medium"   # or "small", "base", "large-v3"
  device: "cpu"     # или "cuda:0" для GPU
```

### 3. (Опционально) Qdrant — база векторов голосов
Если нужен полный пайплайн с кластеризацией и распознаванием по базам голосов:
```bash
docker compose -f docker/docker-compose.yml up -d qdrant
```

### 4. (Опционально) Redis — для блокировок и кэша
```bash
# Встроен в docker-compose.yml
```

---

## Как запустить обработку аудиофайла

### Способ 1: Прямой запуск через Docker (рекомендуемый)

```bash
# Собрать и запустить контейнер с вашим файлом
docker run --rm \
  -v $(pwd):/app/data \
  -e NVIDIA_VISIBLE_DEVICES=all \
  speaker-id-app python -m orchestrator.main \
    --input /app/data/test.wav \
    --output-dir /app/data/output
```

### Способ 2: Локальный запуск (без Docker)

```bash
# Убедитесь, что виртуальное окружение активно
source .venv/bin/activate   # или .venv\Scripts\activate на Windows

# Запустить напрямую
python -m orchestrator.main \
  --input test.wav \
  --output-dir data/output
```

### Способ 3: Через Python-скрипт

Создайте файл `run_test.py`:
```python
from pathlib import Path
from audio.pipeline import ASRPipeline
from config.settings import get_settings

# Загрузка настроек
settings = get_settings()

# Инициализация пайплайна
pipeline = ASRPipeline(settings)

# Обработка файла
result = pipeline.transcribe(
    audio="test.wav",
    output_dir=Path("data/output"),
)

# Вывод результата
for seg in result.segments:
    speaker = f" [{seg.speaker_id}]" if seg.speaker_id else ""
    print(f"{seg.start:.1f}s - {seg.end:.1f}s{speaker}: {seg.text}")

print(f"\nВсего сегментов: {len(result.segments)}")
print(f"Длительность: {result.total_duration:.1f} сек")
```

Запуск: `python run_test.py`

---

## Формат входных аудиофайлов

| Параметр | Требование |
|----------|------------|
| Формат | Любой (поддерживается декодирование torchaudio) |
| Частота дискретизации | Любая (автоматически ресемплируется до 16 кГц) |
| Каналы | Моно или стере |
| Длительность | Нет ограничений, но для длинных файлов (>30 мин) рекомендуется разбивка |

---

## Выходные данные

Результат сохраняется в `data/output/transcription.json`:

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "text": "здравствуйте",
      "speaker_id": null,
      "confidence": 0.95
    },
    {
      "start": 4.1,
      "end": 6.8,
      "text": "добрый день",
      "speaker_id": "spk_0",
      "confidence": 0.87
    }
  ],
  "audio_info": {
    "files_processed": 1,
    "total_duration_seconds": 45.3
  },
  "total_duration": 45.3
}
```

---

## Настройка идентификации говорящих (Speaker-ID)

### Регистрация голоса (Enrollment)

Положите эталонные записи голосов в папку `known_voices/`:
```bash
mkdir -p known_voices
cp admin_voice.wav known_voices/admin.wav
cp manager_ivanov.wav known_voices/manager_ivanov.wav
```

Имя файла = ID говорящего. Система автоматически извлечёт эмбеддинг и сравнит с найденными сегментами.

### Настройка количества спикеров

В `config/config.yaml`:
```yaml
diarization:
  base_similarity_threshold: 0.75   # Порог сходства голосов (меньше = строже)
```

---

## Полезные команды

```bash
# Проверить синтаксис всех файлов
python3 -m py_compile orchestrator/main.py
python3 -m py_compile audio/pipeline.py
python3 -m py_compile config/settings.py

# Запустить все тесты
python3 -m pytest tests/ -v

# Конкретный тест
python3 -m pytest tests/test_asr_pipeline.py -v

# Проверить конфигурацию
python3 -c "from config.settings import get_settings; s = get_settings(); print(s)"

# Запуск с отладочным логированием
SPEAKER_ID__APP__LOG_LEVEL=DEBUG python -m orchestrator.main --input test.wav
```

---

## Возможные проблемы и решения

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError: No module named 'omegaconf'` | `pip install omegaconf` |
| `CUDA out of memory` | Смените `device: "cpu"` в config.yaml или уменьшите размер модели |
| `Model not found` при старте ASR | Первый запуск скачивает модель (~2-6 ГБ). Проверьте интернет. |
| `Qdrant connection refused` | Запустите Qdrant через Docker: `docker compose -f docker/docker-compose.yml up -d qdrant` |
| Пустая транскрипция | Увеличьте длительность аудио или проверьте качество записи |

---

## Структура проекта

```
.
├── audio/                  # Аудио-пайплайн (ASR + VAD)
│   └── pipeline.py         # Основной класс ASRPipeline
├── config/                 # Конфигурация
│   ├── config.yaml         # Все параметры системы
│   └── settings.py         # Валидация настроек Pydantic
├── docker/                 # Docker-контейнер
│   ├── docker-compose.yml  # Оркестрация сервисов
│   └── Dockerfile
├── enrollment/             # Регистрация голосов (следующий этап)
├── orchestrator/           # Оркестратор пайплайна
│   └── main.py            # Точка входа CLI
├── prompts/                # Промпты для LLM
│   └── refine.txt
├── qdrant/                 # Работа с векторной базой
│   └── schema.py
├── tests/                  # Модульные тесты
│   ├── test_asr_pipeline.py
│   ├── test_diarization.py
│   └── conftest.py
├── .env.example            Шаблон переменных окружения
├── requirements.txt        Зависимости Python
└── main.py                 Точка входа (опционально)
```
