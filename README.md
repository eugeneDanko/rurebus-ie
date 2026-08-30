# Извлечение экономических сущностей и отношений из русскоязычных текстов

## Основная задача

Разработать NLP-систему, которая преобразует русскоязычные экономические тексты в структурированные данные:

1. находит значимые сущности и их точные границы в тексте;
2. определяет тип каждой найденной сущности;
3. устанавливает семантические отношения между сущностями;
4. возвращает результат в машиночитаемом формате, например JSON.

Пример целевого результата:

```json
{
  "entities": [
    {"id": "T1", "text": "увеличение", "type": "CMP"},
    {"id": "T2", "text": "объём инвестиций", "type": "MET"}
  ],
  "relations": [
    {"type": "FPS", "arg1": "T1", "arg2": "T2"}
  ]
}
```

## Практическая цель

Конечная цель — создать систему мониторинга русскоязычных экономических новостей. Она должна извлекать компании, учреждения, экономические показатели, изменения, прогнозы и связи между ними.

Возможные применения:

- структурированный поиск по экономическим новостям;
- автоматическое создание карточек событий и компаний;
- отслеживание изменений экономических показателей;
- построение графа экономических сущностей и отношений;
- подготовка структурированного контекста для аналитической или RAG-системы;
- последующее добавление entity-level sentiment analysis.

## Запуск NER baseline

Baseline реализован как устанавливаемый пакет: RuBERT encoder + линейная
17-классовая BIO-голова. Ноутбуки содержат только сценарии запуска, а
токенизация, выравнивание BRAT offsets, обучение, метрики и inference находятся
в `src/rurebus_ie/`.

В Google Colab последовательно выполните:

1. `00_download_rurebus.ipynb` — если raw-данные ещё не загружены;
2. `02_preprocessing.ipynb` — создаёт manifest и train/validation/test;
3. `03_train_ner.ipynb` — обучает и сохраняет лучший validation-checkpoint;
4. `04_test_ner.ipynb` — считает финальные test-метрики и сохраняет предсказания.
5. `05_ner_error_analysis.ipynb` — регистрирует `B0` и анализирует ошибки на validation.

Каждый новый Colab runtime является чистой средой. Первые ячейки ноутбуков
автоматически запускают `colab_bootstrap.py`: проверяют `PROJECT_DIR`, делают
editable-установку в текущий Python kernel и сразу проверяют импорт
`rurebus_ie`. Перезапуск runtime после установки не требуется.

Те же этапы доступны из командной строки после `pip install -e .`:

```bash
rurebus-preprocess --data-root rurebus_data
rurebus-train-ner configs/experiments/ner_baseline_v1.yaml
rurebus-test-ner configs/experiments/ner_baseline_v1.yaml
rurebus-analyze-ner configs/experiments/ner_baseline_v1.yaml
```

Для обучения рекомендуется GPU. Результаты сохраняются в
`results/ner_baseline/ner_baseline_v1/`: конфигурация запуска, история эпох,
лучший checkpoint, strict entity-level метрики и `predictions.jsonl`.

После регистрации `B0` создаются `baseline_record.json` с SHA-256 артефактов и
`.baseline_locked`. Обучение не перезапишет зарегистрированный output-каталог:
для следующего эксперимента необходимо задать новое имя и `output_dir`.
Validation error analysis сохраняется в
`error_analysis/validation/` и не использует test для настройки модели.

RuREBus содержит не биржевые новости, а стратегические документы и отчёты государственных органов. Поэтому он используется для построения и проверки базового NER/RE-пайплайна. Для переноса системы на экономические новости позднее потребуется отдельный небольшой корпус новостей с ручной разметкой и расширенной схемой сущностей.

## Данные

### RuREBus

Основной датасет первого этапа — **RuREBus (Russian Relation Extraction for Business)**.

- Официальный репозиторий и данные: <https://github.com/dialogue-evaluation/RuREBus>
- Альтернативная карточка на Hugging Face: <https://huggingface.co/datasets/iluvvatar/RuREBus>
- Статья с описанием корпуса: <https://arxiv.org/abs/2010.15939>

Корпус содержит русскоязычные региональные отчёты, стратегические планы и прогнозы Минэкономразвития. Разметка хранится в формате BRAT: исходному `.txt`-документу соответствует `.ann`-файл с символьными границами сущностей и отношениями.

В размеченной части представлены восемь типов сущностей:

| Тип | Значение |
|---|---|
| `MET` | Измеримый показатель или объект сравнения |
| `ECO` | Экономический объект или объект инфраструктуры |
| `BIN` | Отдельное действие или бинарная характеристика |
| `CMP` | Изменение или сравнительная характеристика |
| `QUA` | Качественная характеристика |
| `ACT` | Мероприятие, проект или деятельность |
| `INST` | Учреждение, структура или организация |
| `SOC` | Социальный объект или явление |

Типы отношений:

- `GOL` — абстрактная цель;
- `TSK` — конкретная задача;
- `PNG`, `PNT`, `PPS` — прошлое состояние с отрицательной, нейтральной или положительной оценкой;
- `NNG`, `NNT`, `NPS` — настоящее состояние с отрицательной, нейтральной или положительной оценкой;
- `FNG`, `FNT`, `FPS` — будущее состояние или прогноз с отрицательной, нейтральной или положительной оценкой.

В relation classifier дополнительно используется технический класс `NO_RELATION` для пар, между которыми нет размеченной связи.

### Зафиксированные выводы EDA

EDA выполнен в `01_rurebus_eda.ipynb` по распакованным данным. Его основные результаты считаются контекстом проекта:

#### Состав и единица данных

- Один пример — пара файлов с одинаковым именем: `.txt` содержит исходный текст, `.ann` — BRAT-разметку.
- Сущность задаётся записью `T...`: ID, тип, символьные границы и точный фрагмент текста. Отношение задаётся записью `R...`: ID, тип и направленные аргументы `Arg1`/`Arg2`, ссылающиеся на сущности.
- Файлы с суффиксами `_part_N` являются фрагментами одного исходного документа, а не независимыми документами. Поэтому при разбиении они объединяются по `source_id`, полученному удалением этого суффикса.
- Все проверенные размеченные пары корректно читаются BRAT-парсером; символьные границы совпадают с текстом, ссылки отношений разрешаются, round-trip `parse → serialize → parse` проходит успешно.
- В корпусе не обнаружены разрывные сущности, хотя парсер технически поддерживает несколько интервалов.

#### Объём размеченных данных

| Набор | Фрагменты | Группы исходных документов | Сущности | Отношения |
|---|---:|---:|---:|---:|
| Три train-партии до удаления дублей | 188 | — | — | — |
| Объединённый train после удаления точных дублей | 182 | 108 | 52 063 | 12 331 |
| Официальный `test_full` | 30 | 18 | 8 457 | 2 038 |

В train-партиях найдено 6 групп текстовых дублей, поэтому исключено 6 повторных копий. В двух группах одинаковому тексту соответствуют различающиеся `.ann`; это фиксируется в `duplicates.csv`, а каноническая версия выбирается с приоритетом `train_3 → train_2 → train_1`.

После preprocessing каноническое разбиение имеет следующий вид:

| Split | Фрагменты | Группы исходных документов | Сущности | Отношения |
|---|---:|---:|---:|---:|
| `train` | 143 | 86 | 41 771 | 9 799 |
| `validation` | 39 | 22 | 10 292 | 2 532 |
| `test` | 30 | 18 | 8 457 | 2 038 |

Пересечений между split по `source_id` и SHA-256 текста нет. В validation присутствуют все 8 типов сущностей и все 11 типов отношений.

#### Устройство тестовых данных

- `test_raw` — первоначально опубликованные тексты без `.ann`.
- `test_ner_only` — 544 текста с разметкой сущностей, всего 89 879 сущностей.
- `test_full` — 30 текстов с сущностями и отношениями; он является подмножеством `test_ner_only`, причём наборы сущностей совпадают.
- После исключения `test_full` остаётся 514 дополнительных NER-размеченных текстов. Их нельзя включать в обучение при официальном сравнении на соответствующем тесте без явного изменения протокола эксперимента.
- Для воспроизводимого общего NER+RE benchmark проект использует `test_full` только как финальный test и не применяет его при выборе модели и гиперпараметров.

#### Практический смысл данных для моделей

- Разметка непосредственно поддерживает две задачи: извлечение 8 типов сущностей (NER) и классификацию 11 типов направленных отношений (RE).
- Распределения сущностей и особенно отношений несбалансированы; оценка должна приводиться по классам и включать macro-F1, а не ограничиваться accuracy или micro-F1.
- Размеченные отношения разрежены: среди всех теоретически возможных пар сущностей положительных связей около 0,08 %. Поэтому нельзя создавать все пары без ограничений — для RE нужны разумная генерация кандидатов, `NO_RELATION` и контролируемый negative sampling.
- Большинство отношений локальны по тексту, поэтому кандидатов целесообразно ограничивать предложением или небольшим контекстным окном, сохраняя возможность отдельно проверить более дальние связи.
- Исходную BRAT-разметку следует сохранять без нормализации и изменения текста: любые изменения символов нарушат записанные offsets.
- EDA подтверждает выбор двухэтапного baseline: сначала RuBERT для NER, затем R-BERT для RE на размеченных сущностях; совместную span-based модель имеет смысл сравнивать с ним позднее.

## Структура проекта и слои данных

Целевая структура проекта как устанавливаемого Python-пакета:

```text
NER_RuREBus_project/
├── README.md
├── pyproject.toml
├── .gitignore
│
├── 00_download_rurebus.ipynb
├── 01_rurebus_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_train_ner.ipynb
├── 04_test_ner.ipynb
├── 05_ner_error_analysis.ipynb
├── 06_review_corrections.ipynb
├── 07_build_corrected_v1.ipynb
├── 08_train_corrected_baseline.ipynb
├── 09_compare_dataset_versions.ipynb
├── colab_bootstrap.py
├── docs/
│   └── RuREBus_glossary.docx
│
├── configs/
│   ├── data/
│   │   ├── rurebus.yaml
│   │   ├── rurebus_corrected_v1.yaml
│   │   └── corrections/
│   │       └── rurebus_corrected_v1.csv
│   ├── models/
│   │   ├── ner_baseline.yaml
│   │   ├── rbert.yaml
│   │   └── span_joint.yaml
│   └── experiments/
│       ├── ner_baseline_v1.yaml
│       ├── ner_baseline_corrected_v1.yaml
│       ├── ner_baseline_corrected_v1_seed17.yaml
│       └── ner_baseline_corrected_v1_seed73.yaml
│
├── src/
│   └── rurebus_ie/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── brat_parser.py
│       │   ├── preprocessing.py
│       │   ├── conflict_resolution.py
│       │   ├── dataset_versioning.py
│       │   ├── ner_dataset.py
│       │   └── collators.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── ner_baseline.py
│       │   ├── rbert.py
│       │   └── span_joint.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── ner_trainer.py
│       │   ├── ner_experiment.py
│       │   ├── artifacts.py
│       │   ├── relation_trainer.py
│       │   └── callbacks.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── ner_metrics.py
│       │   ├── ner_error_analysis.py
│       │   ├── relation_metrics.py
│       │   └── evaluator.py
│       └── inference/
│           ├── __init__.py
│           ├── ner_pipeline.py
│           └── relation_pipeline.py
│
├── rurebus_data/
│   ├── RuREBus/                  # raw-слой: исходный репозиторий и разметка
│   │   ├── train_data/
│   │   ├── test_data/
│   │   ├── eval_scripts/
│   │   └── markup_instruction.pdf
│   ├── processed/                # original_processed_v1
│   │   ├── train/
│   │   ├── validation/
│   │   ├── test/
│   │   ├── manifest.csv
│   │   ├── duplicates.csv
│   │   └── preprocessing_report.json
│   └── versions/
│       └── corrected_v1/         # неизменяемая версия с принятыми правками
│           ├── train/
│           ├── validation/
│           ├── test/
│           ├── manifest.csv
│           ├── dataset_report.json
│           └── preprocessing_report.json
│
├── results/
│   └── <model_name>/
│       └── <experiment_id>/
│           ├── config.yaml
│           ├── train_metrics.json
│           ├── validation_metrics.json
│           ├── test_metrics.json
│           ├── predictions.jsonl
│           ├── history.csv
│           ├── baseline_record.json
│           ├── validation_predictions.jsonl
│           ├── error_analysis/validation/
│           └── checkpoints/
│               └── best/
│
└── tests/
    ├── test_brat_parser.py
    ├── test_conflict_resolution.py
    ├── test_dataset_versioning.py
    ├── test_preprocessing.py
    ├── test_ner_dataset.py
    └── test_ner_model.py
```

Ноутбуки намеренно остаются в корне проекта: они представляют последовательные этапы исследования и экспериментов и сразу видны при открытии проекта в Google Drive или Colab. При этом ноутбуки содержат только сценарии запуска, визуализацию и анализ результатов. Реализация подготовки данных, моделей, обучения и оценки находится в устанавливаемом пакете.

`src/` является техническим контейнером исходного кода и не входит в путь Python-импорта. Непосредственный пакет называется `rurebus_ie`, поэтому корректные импорты выглядят так:

```python
from rurebus_ie.data.brat_parser import load_brat_document
from rurebus_ie.models.ner_baseline import build_rubert_token_classifier
from rurebus_ie.training.ner_trainer import NerTrainer
```

Импорты вида `from src...` не используются.

### Ответственность каталогов

- `configs/` хранит декларативные YAML-конфигурации данных, моделей и экспериментов. Фактический состав split фиксируется в `manifest.csv`, а не дублируется вручную в YAML.
- `src/rurebus_ie/data/` содержит чтение BRAT, preprocessing, model-specific datasets и collators.
- `src/rurebus_ie/models/` содержит только архитектуры, `forward()` и model-specific loss.
- `src/rurebus_ie/training/` управляет эпохами, оптимизаторами, validation, early stopping и сохранением checkpoint.
- `src/rurebus_ie/evaluation/` рассчитывает метрики и формирует отчёты.
- `src/rurebus_ie/inference/` восстанавливает сущности и отношения из предсказаний обученных моделей.
- `results/` хранит отдельный каталог для каждого запуска, чтобы эксперименты не перезаписывали друг друга. Большие checkpoints и генерируемые результаты не коммитятся в Git.
- `tests/` содержит проверки парсинга, preprocessing, model-specific подготовки данных и контрактов моделей.

Модельный класс не должен одновременно реализовывать архитектуру, полный training loop, оценку и сохранение отчётов. Эти обязанности разделяются между `models`, `training`, `evaluation` и `inference`.

`rurebus_data/RuREBus/` рассматривается как неизменяемый raw-слой. Исходные `.txt`, `.ann` и ZIP-архивы не редактируются. Канонический исходный split сохраняется в `rurebus_data/processed/`, а исправленные неизменяемые версии — в `rurebus_data/versions/<version>/`.

### BRAT-парсер

`src/rurebus_ie/data/brat_parser.py` реализует модель-независимое чтение фактического BRAT standoff-формата RuREBus:

- строки `T...` преобразуются в сущности с типом, текстом и символьными интервалами;
- строки `R...` преобразуются в отношения `Arg1`/`Arg2`;
- проверяются типы, уникальность ID, символьные границы, совпадение `text[start:end]` и ссылки отношений;
- поддерживается обратная сериализация ANN и round-trip проверка.

Парсер основан только на стандартной библиотеке Python. Для него достаточно пар `.txt + .ann`, которые уже присутствуют во всех трёх train-партиях и в `test_full`. Отсутствие `.ann` в исходном `test` является ожидаемым: этот набор был опубликован без разметки.

### Канонический preprocessing

`src/rurebus_ie/data/preprocessing.py` выполняет только преобразования, не зависящие от модели:

1. загружает и валидирует три train-партии;
2. проверяет обратимость BRAT-разметки;
3. объединяет партии в единый логический train;
4. удаляет точные дубли текста по SHA-256 с приоритетом `train_part_3 → train_part_2 → train_part_1`;
5. группирует фрагменты одного исходного документа по имени без суффикса `_part_N`;
6. формирует воспроизводимый train/validation split с сохранением редких типов отношений;
7. использует официальный `test_full` как закрытый test;
8. копирует выбранные BRAT-пары без изменения содержимого;
9. создаёт manifest, отчёт о дублях и JSON-отчёт проверок.

`manifest.csv` содержит идентификатор документа и исходной группы, split, исходную партию, пути raw/processed, контрольные суммы, длины и распределения типов сущностей и отношений.

На этом этапе намеренно не выполняются нормализация текста, BIO/BILOU-конвертация, токенизация RuBERT, оконная нарезка, генерация `NO_RELATION` и negative sampling. Эти операции зависят от выбранной модели и относятся к следующему слою подготовки данных.

### Версионирование исправленной разметки

Предложения по исправлениям хранятся в `configs/data/corrections/rurebus_corrected_v1.csv`. Каждая строка имеет один из статусов:

- `ACCEPTED` — применяется при сборке;
- `REJECTED` — явно отклонена;
- `REVIEW_REQUIRED` — не применяется до ручной контекстной ревизии.

`06_review_corrections.ipynb` объединяет решения с полным реестром контекстов. `07_build_corrected_v1.ipynb` проверяет SHA-256 родительского корпуса и атомарно создаёт `rurebus_data/versions/corrected_v1`. Существующая версия не перезаписывается: после изменения решений необходимо создать `corrected_v2`.

Эквивалентные CLI-команды:

```bash
rurebus-build-version build \
  --source-root rurebus_data/processed \
  --output-root rurebus_data/versions/corrected_v1 \
  --corrections configs/data/corrections/rurebus_corrected_v1.csv \
  --source-integrity-manifest results/data_audit/rurebus_conflict_resolution_v1/file_integrity.csv \
  --dataset-version corrected_v1 \
  --parent-version original_processed_v1

rurebus-build-version validate rurebus_data/versions/corrected_v1
```

В `dataset_report.json` фиксируются fingerprint родителя и результата, SHA-256 файла решений, число применённых исправлений, распределения классов и политика оценки. Test в `corrected_v1` является внутренним audited gold: его метрику нельзя напрямую сравнивать с официальной метрикой на исходном test.

`08_train_corrected_baseline.ipynb` повторяет одну и ту же архитектуру и гиперпараметры на seed 42, 17 и 73. Каждый checkpoint оценивается дважды: на внутреннем `corrected_v1/test` и на неизменённом `original/test`. `09_compare_dataset_versions.ipynb` использует исходный test для сопоставления с B0, а corrected test показывает качество относительно очищенного внутреннего gold.

Запуск в Colab выполняется через `02_preprocessing.ipynb`. После установки пакета эквивалентная CLI-команда запускается как Python-модуль:

```bash
python -m rurebus_ie.data.preprocessing \
  --data-root /content/drive/MyDrive/NER_RuREBus_project/rurebus_data \
  --validation-size 0.2 \
  --seed 42 \
  --search-trials 5000 \
  --overwrite
```

### Использование в Google Colab

После подключения Google Drive нужно перейти в корень проекта и один раз за сессию установить пакет в editable-режиме:

```python
from google.colab import drive

drive.mount("/content/drive")
%cd /content/drive/MyDrive/NER_RuREBus_project
%pip install -e .
```

После этого модули импортируются из любого ноутбука без абсолютных путей и без изменения `sys.path`:

```python
from rurebus_ie.data.brat_parser import load_brat_document
from rurebus_ie.models.ner_baseline import build_rubert_token_classifier
```

Editable-установка означает, что изменения файлов внутри `src/rurebus_ie/` становятся доступны без переустановки пакета. Перезапуск установки требуется только после изменения зависимостей или метаданных в `pyproject.toml`.

Пути к данным и результатам берутся из YAML-конфигураций и разрешаются относительно корня проекта. Они не должны быть захардкожены внутри моделей или training-классов.

### Зафиксированная ошибка импорта в Colab

Известный инцидент проекта:

```text
ModuleNotFoundError: No module named 'rurebus_ie.data.conflict_resolution'
```

Ошибка возникла после обновления локального проекта, когда Google Drive содержал
старую копию `src/rurebus_ie/`, а kernel Colab продолжал хранить ранее
импортированные модули. `pip install` стороннего пакета эту проблему не решает:
Colab должен использовать код именно из текущего `PROJECT_DIR`.

Обязательные правила для всех ноутбуков:

1. Сначала выполняется первая bootstrap-ячейка, только затем импорты из
   `rurebus_ie`.
2. В Google Drive должны присутствовать актуальные файлы
   `src/rurebus_ie/data/conflict_resolution.py` и
   `src/rurebus_ie/data/dataset_versioning.py`.
3. Используется только `colab_bootstrap.bootstrap_project(PROJECT_DIR)`. Ручные
   вставки произвольных каталогов в `sys.path` не используются.
4. После обновления кода достаточно повторно выполнить bootstrap-ячейку:
   bootstrap удаляет старые `rurebus_ie.*` из `sys.modules`, ставит текущий
   `src/` первым в `sys.path`, выполняет editable install и проверяет шесть
   обязательных модуля. Перезапуск runtime не требуется.
5. Если исходный файл отсутствует в Google Drive, bootstrap останавливается до
   запуска эксперимента и печатает ожидаемый абсолютный путь. В таком случае
   нужно обновить копию проекта, а не устанавливать одноимённый пакет из PyPI.

Успешная bootstrap-проверка заканчивается сообщениями:

```text
rurebus_ie imported successfully: .../NER_RuREBus_project/src/rurebus_ie/__init__.py
Required modules verified: 6
```

Для синхронизации версии `corrected_v1` используется архив
`colab_project_update_corrected_v1.zip`: он содержит актуальный пакет, конфиги,
ноутбуки и архив датасета. После его распаковки в корень проекта нужно заново
запустить первую ячейку `07_build_corrected_v1.ipynb`.

## Иерархический Span NER с двухэтапным обучением

`16_train_hierarchical_span_ner.ipynb` запускает curriculum-вариант Span NER,
не изменяя внешнюю RuREBus-схему. Предсказания и итоговые метрики по-прежнему
используют восемь исходных имён `ACT`, `BIN`, `CMP`, `QUA`, `ECO`, `SOC`, `MET`,
`INST`.

Внутренние группы задаются в
`configs/models/hierarchical_span_ner.yaml`: `ACT+BIN`, `CMP+QUA`, `ECO+SOC`,
а `MET` и `INST` остаются отдельными. На первом этапе модель оптимизирует
классификацию этих групп и supervised contrastive loss по группам. На втором
этапе оптимизируются исходные классы, auxiliary loss суперклассов и supervised
contrastive loss уже по исходным меткам. Оптимизатор и scheduler между этапами
создаются заново.

Конфигурация по умолчанию выполняет 5 coarse-эпох и 15–20 fine-эпох. Лучший
checkpoint выбирается только по fine strict entity micro-F1 и сохраняется один
раз. Эквивалентный CLI-запуск:

```bash
rurebus-train-hierarchical-span-ner \
  configs/experiments/hierarchical_span_ner_corrected_v1.yaml
```

После обучения `17_calibrate_hierarchical_span_threshold.ipynb` выбирает
единый confidence threshold только на validation и только для fine-head. Он
делает один encoder inference-проход без передачи gold-меток в модель, поэтому
fine/coarse/contrastive losses при калибровке не вычисляются, и сохраняет
`threshold_calibration.csv/json` рядом с run; test при выборе порога не
используется. Обычный `11_calibrate_span_threshold.ipynb` предназначен для
неиерархической `SpanNerModel` и к этому checkpoint не применяется.

## Основные архитектуры

### 1. RuBERT Token Classification

```text
токенизатор → RuBERT → линейный классификатор → BIO-декодер
```

Архитектура используется как основной NER baseline.

- Токенизатор разбивает текст на subword-токены и сохраняет их символьные позиции.
- RuBERT создаёт контекстный вектор каждого токена.
- Линейный классификатор назначает токену BIO-метку, например `B-MET`, `I-MET` или `O`.
- BIO-декодер объединяет последовательности токенов в сущности с типом и точными границами.

### 2. R-BERT для классификации отношений

```text
генератор пар → маркеры сущностей → RuBERT → pooling → классификатор отношений
```

Архитектура используется после NER-модели.

- Генератор пар формирует кандидатов из найденных сущностей.
- Маркеры `[E1]...[/E1]` и `[E2]...[/E2]` показывают, какую пару требуется классифицировать.
- RuBERT кодирует сущности вместе с окружающим контекстом.
- Pooling объединяет представление предложения и представления двух сущностей.
- Классификатор выбирает один из типов отношений либо `NO_RELATION`.

### 3. Совместная Span-based RuBERT-модель

```text
токенизатор → RuBERT → генератор span → классификатор сущностей
                                ↓
                   генератор пар → классификатор отношений
```

Это продвинутая архитектура для следующего этапа.

- Один RuBERT создаёт общие контекстные представления текста.
- Генератор span создаёт кандидаты на сущности как интервалы `[start, end]`.
- Классификатор сущностей назначает каждому span тип или `NONE`.
- Из найденных span формируются пары.
- Отдельная голова определяет отношения между парами сущностей.
- Обе задачи обучаются совместно через общий NER/RE loss.

## План развития архитектуры

1. Реализовать загрузку и валидацию BRAT-разметки RuREBus.
2. Подготовить BIO-представление данных.
3. Обучить `RuBERT Token Classification` и получить воспроизводимый NER baseline.
4. Обучить R-BERT на правильных, заранее размеченных сущностях.
5. Соединить NER и RE в двухэтапный end-to-end pipeline.
6. Реализовать совместную span-based модель как улучшение baseline.
7. Разработать расширенную схему для экономических новостей.
8. Разметить небольшой новостной корпус и выполнить domain adaptation.

## Locked protocol `global_v1` и полный NER → RE pipeline

Предыдущий RuREBus test уже использовался при сравнении NER-моделей и анализе
ошибок. Поэтому он не считается независимым финальным test. Notebook
`18_build_global_protocol.ipynb` один раз создаёт manifest-only протокол:

```text
старые train + validation
├── train       ≈ 70%
├── validation  ≈ 15%
└── global_test ≈ 15%  (только документы из прежнего train)

старый test → legacy_test
```

Группировка выполняется по `source_id`, точные дубли текста между split
запрещены. Manifest и отчёт защищены SHA-256. BRAT-файлы не копируются, поэтому
новый протокол почти не занимает место в Google Drive. Старые NER checkpoints
нельзя честно тестировать на `global_test`: они уже обучались на части этих
документов. Для нового протокола обе модели обучаются заново.

Последовательность запуска:

1. `18_build_global_protocol.ipynb` — резервирует split;
2. `19_train_hierarchical_span_ner_global_v1.ipynb` — переобучает NER;
3. `20_calibrate_hierarchical_span_ner_global_v1.ipynb` — выбирает NER threshold
   на validation и сохраняет validation-предсказания;
4. `21_train_relation_classifier_global_v1.ipynb` — обучает R-BERT на gold
   train-сущностях;
5. `22_calibrate_relation_classifier_global_v1.ipynb` — калибрует компонентный
   RE threshold на gold validation-сущностях;
6. `23_calibrate_end_to_end_pipeline_global_v1.ipynb` — калибрует RE threshold
   на NER-предсказаниях validation;
7. `24_final_global_test.ipynb` — один раз оценивает NER, RE на gold-сущностях и
   полный NER→RE pipeline.

Relation classifier использует `[CLS] + [E1] + [E2]` pooling поверх RuBERT и
11 независимых sigmoid-выходов. Это намеренный multi-label вариант R-BERT:
в корпусе есть пара, размеченная одновременно `NNG` и `TSK`; `NO_RELATION`
представляется нулевым вектором. Кандидаты — направленные пары непересекающихся
сущностей с разрывом не более 128 символов. Ограничение покрывает 99.35% всех
gold-отношений исходного корпуса, а непокрытые отношения остаются false negative
и отражаются в `candidate_recall`.

Обучающие notebooks сохраняют промежуточные веса в `/content`. В Google Drive
через `persist_best_run()` переносится только один best checkpoint, tokenizer,
конфигурация и метрики. Кэши Hugging Face также направлены в `/content`, а не в
папку проекта.

## Границы первого этапа

На первом этапе не требуются база знаний, мешок слов или собственные embeddings. Достаточно:

- RuREBus;
- предобученного RuBERT и соответствующего токенизатора;
- BRAT-парсера;
- BIO-схемы;
- списка типов сущностей и отношений;
- функций обучения, декодирования и оценки.

База компаний, тикеров и алиасов понадобится позднее для entity linking и нормализации названий, но не для первоначального NER/RE baseline.
