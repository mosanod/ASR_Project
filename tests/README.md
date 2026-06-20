# Speaker-ID ASR Pipeline — Тесты

## Обзор

Этот репозиторий содержит модульные тесты для компонентов системы распознавания речи с идентификацией говорящего.

## Структура тестов

### `test_asr_pipeline.py` — Тесты ASR (распознавание речи)
- **test_initialization** — Проверка инициализации pipeline
- **test_settings_loading** — Загрузка настроек из конфигурации
- **test_overlap_calculation** — Вычисление перекрытия между сегментами
- *(дополнительные тесты могут быть добавлены)*

### `test_diarization.py` — Тесты диаризации (идентификация говорящих)
#### SpeakerEmbeddingExtractor
- **test_extraction_returns_correct_shape** — Проверка размерности эмбеддингов (192 dim для TitaNet)
- **test_extraction_handles_short_audio** — Обработка коротких аудиофрагментов
- **test_extraction_returns_normalized_embeddings** — Нормализация векторов

#### SpeakerClustering
- **test_cluster_single_embedding** — Кластеризация одиночного эмбеддинга
- **test_cluster_similar_embeddings** — Объединение похожих говорящих
- **test_cluster_different_embeddings** — Разделение разных говорящих
- **test_cluster_empty_embeddings** — Обработка пустого ввода

#### DiarizationEngine
- **test_process_empty_segments** — Обработка пустых сегментов
- **test_process_single_segment** — Обработка одного сегмента
- **test_process_multiple_segments_same_speaker** — Группировка сегментов одного говорящего
- **test_process_multiple_segments_different_speakers** — Разделение разных говорящих
- **test_get_speaker_count** — Оценка количества говорящих

#### DiarizeAudio (удобная функция)
- **test_diarize_audio_returns_dict** — Проверка возвращаемого типа

## Запуск тестов

### Все тесты
```bash
python3 -m pytest tests/ -v
```

### Конкретный файл
```bash
python3 -m pytest tests/test_asr_pipeline.py -v
python3 -m pytest tests/test_diarization.py -v
```

### Конкретный тест
```bash
python3 -m pytest tests/test_diarization.py::TestSpeakerClustering::test_cluster_similar_embeddings -v
```

### Покрытие кода
```bash
python3 -m pytest tests/ --cov=audio --cov-report=html
```

## Требования для запуска тестов

- Python 3.12+
- `pytest` (установлен через `requirements.txt`)
- `numpy`
- `pydantic-settings`

## Примечания

- Тесты используют мок-объекты для зависимостей от PyTorch и внешних моделей
- Для реальных тестов с аудиофайлами используйте fixtures в `conftest.py`
- Все тесты должны выполняться за < 5 секунд (быстрые unit-тесты)
