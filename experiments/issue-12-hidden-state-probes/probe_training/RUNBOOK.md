# Qwen3-4B: запуск instruction-specific probes

Для нового варианта с общим attention pooling используйте
[ATTENTION_RUNBOOK.md](ATTENTION_RUNBOOK.md). Ему нужны отдельные hidden states
токенов; прежних усреднённых признаков недостаточно.

Все команды выполняются из корня `memWMLM`. Используется существующая база
`experiments/issue-12-hidden-state-probes/artifacts/db`: создавать новую не нужно.
Признаки необходимо извлечь именно из Qwen3-4B; признаки Qwen3-1.7B не подходят.

Результат запуска — **56 отдельных классификаторов**:
7 вопросов × 4 инструкции (`none`, `imagine`, `track_people`, `track_mary`) ×
2 порядка фактов (`canonical`, `shuffled`). `none` означает отсутствие инструкции.
Каждый классификатор выбирает собственные слой, pooling и L2 на validation
своего условия. Подробности протокола и ограничения описаны в [README](README.md).

## 1. Проверить окружение

Используйте существующее `.venv`. Если окружения ещё нет и `uv` установлен:

```bash
uv sync --extra dev
```

Зафиксируйте версию кода и проверьте GPU:

```bash
git rev-parse HEAD
git status --short
nvidia-smi
.venv/bin/python -c "import torch, transformers; print('torch:', torch.__version__, 'transformers:', transformers.__version__); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))"
```

BF16-веса 4B занимают около 8 GB, дополнительно нужна память для активаций.
Команды ниже используют GPU 0 и batch size 4. При нехватке памяти уменьшите
`--batch-size` до 2 или 1; при выборе другого GPU замените `CUDA_VISIBLE_DEVICES=0`.
Обучение линейных probes по умолчанию выполняется на CPU.

Проверить код эксперимента:

```bash
.venv/bin/python -m pytest -q \
  tests/test_spatial_probe_dataset.py \
  tests/test_hidden_feature_extraction.py \
  tests/test_linear_hidden_probes.py \
  tests/test_instruction_specific_probes.py
```

## 2. Проверить существующую базу

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/probe_training/train.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --audit-only
```

Аудит проверяет хеши, отсутствие пересечения групп/эпизодов между split,
полноту парных условий, совпадение состояний и фактов, наличие классов.
Модель при этом не загружается, обучение не запускается.

Ожидаемые размеры:

| Split | Все строки | Независимые группы | Строк на одну инструкцию × порядок |
|---|---:|---:|---:|
| train | 3264 | 17 | 408 |
| validation | 576 | 3 | 72 |
| test | 768 | 4 | 96 |

Если база отсутствует на другой машине, перенесите всю папку `artifacts/db`
вместе с `manifest.json`. Для текущего рабочего каталога генерацию пропускаем.

## 3. Проверить закреплённую модель

Используется та же ревизия, что и в full-context QA для 4B:
`1cfa9a7208912126459214e8b04321603b3df60c`.
Если её ещё нет в локальном кеше, скачайте:

```bash
.venv/bin/hf download Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c
```

Проверить конфигурацию из кеша:

```bash
.venv/bin/python - <<'PY'
from transformers import AutoConfig
config = AutoConfig.from_pretrained(
    'Qwen/Qwen3-4B',
    revision='1cfa9a7208912126459214e8b04321603b3df60c',
    local_files_only=True,
)
print('model_type:', config.model_type)
print('layers:', config.num_hidden_layers, 'hidden_size:', config.hidden_size)
PY
```

Экстрактор загружает локальные веса. Модель заморожена, обрабатывается только
ситуация с соответствующей инструкцией, без вопроса и без chat template.

## 4. Извлечь признаки Qwen3-4B

Папка назначения не должна существовать. Создавать её вручную не нужно.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-hidden-state-probes/probe_training/artifacts/features-qwen3-4b \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --layers 4,8,12,16,20,24,28,last \
  --device cuda \
  --model-dtype bfloat16 \
  --output-dtype float16 \
  --batch-size 4 \
  --max-length 256
```

`last` автоматически разрешается в последний слой модели. Сохраняются
`last_token`, `mean_pool`, `sentence_end_mean` для всех выбранных слоёв.
Если нужен короткий пробный запуск, используйте **другую** output-папку и
`--max-examples-per-split 8`. Такие неполные признаки нельзя передавать в полное
обучение. Полный запуск выполняется без этого ограничения.

Проверить завершённый manifest:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('experiments/issue-12-hidden-state-probes/probe_training/artifacts/features-qwen3-4b')
m = json.loads((p / 'manifest.json').read_text())
assert m['model']['name'] == 'Qwen/Qwen3-4B'
assert m['model']['requested_revision'] == '1cfa9a7208912126459214e8b04321603b3df60c'
assert m['model']['trainable_parameters'] == 0
assert m['model']['eval_mode']
assert not m['tokenization']['chat_template_applied']
for split, count in {'train': 3264, 'validation': 576, 'test': 768}.items():
    assert m['splits'][split]['examples'] == count
print(m['model'])
print(m['features']['layer_indices'])
print('Feature manifest OK; shard hashes and row IDs are checked by the trainer.')
PY
```

## 5. Обучить отдельные классификаторы

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/probe_training/train.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --feature-dir experiments/issue-12-hidden-state-probes/probe_training/artifacts/features-qwen3-4b \
  --output-dir experiments/issue-12-hidden-state-probes/probe_training/artifacts/probes-qwen3-4b \
  --poolings last_token,mean_pool,sentence_end_mean \
  --l2-values 0,1e-5,1e-4,1e-3 \
  --max-iter 100 \
  --device cpu \
  --threads 1 \
  --bootstrap-samples 2000 \
  --seed 12012
```

Для каждого из 56 классификаторов перебираются pooling × слой × L2.
Стандартизация и веса обучаются только на train. Выбор производится по
validation balanced accuracy данного вопроса; при равенстве — обычная accuracy,
более сильный L2, затем первый кандидат. Бюджет LBFGS фиксирован: 100 итераций.
Test не участвует в выборе. Переобучения на train+validation нет.

В ходе работы печатаются строки `Locked instruction/order/question` с выбранной
конфигурацией. Затем считаются парные bootstrap-интервалы. Не изменяйте сетку
параметров по результатам test.

## 6. Проверить и прочитать результаты

В `probe_training/artifacts/probes-qwen3-4b/` должны появиться:

- 56 файлов `instruction--order--question.pt`: веса, стандартизация, классы и
  выбранные параметры каждого probe;
- `report.json`: все validation-кандидаты, выбранные параметры, validation/test
  метрики, shuffled-label контроль и парные эффекты;
- `test_predictions.jsonl`: предсказания классов для аудита.

Вывести основные результаты:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('experiments/issue-12-hidden-state-probes/probe_training/artifacts/probes-qwen3-4b')
r = json.loads((p / 'report.json').read_text())
assert len(list(p.glob('*.pt'))) == 56
assert len(r['cells']) == 8
assert all(len(cell['selection']) == 7 for cell in r['cells'].values())
assert sum(1 for _ in (p / 'test_predictions.jsonl').open()) == 768
for name, cell in r['cells'].items():
    test = cell['evaluation']['test']
    control = cell['shuffled_label_control_test']
    print(name, 'macro:', test['mean_macro_accuracy'],
          'joint:', test['joint_exact_state_accuracy'],
          'shuffled-label macro:', control['mean_macro_accuracy'])
    for question, selection in cell['selection'].items():
        print(' ', question, selection['selected'])
for order in ('canonical', 'shuffled'):
    name = f'track_mary-minus-none/{order}'
    effect = r['instruction_effects'][name]
    print(name)
    for key in ('mary_relations', 'bob_john_relations',
                'selectivity_difference_in_differences'):
        print(' ', key, effect[key])
PY
```

Поля отчёта:

| Поле | Содержание |
|---|---|
| `cells["none/canonical"]` | Банк без инструкции, canonical факты |
| `cells[...].selection[question]` | Индивидуальный выбор и все кандидаты |
| `cells[...].evaluation.test.per_probe` | Test accuracy и balanced accuracy каждого вопроса |
| `instruction_effects` | Каждая инструкция минус `none`, отдельно по порядку фактов |
| `fact_order_effects` | Shuffled минус canonical для каждой инструкции |

В `instruction_effects` значение `delta` — разность долей, например `0.05`
означает +5 процентных пунктов. `paired_group_bootstrap_95ci` — интервал
разности, а не два независимых интервала отдельных accuracy.

Основная selective-effect проверка для `track_mary`:

- `mary_relations`: изменение четырёх направленных отношений с Mary;
- `bob_john_relations`: изменение двух направленных отношений Bob–John;
- `selectivity_difference_in_differences`: первое изменение минус второе.

`leftmost_person` выводится отдельно как `probe/leftmost_person`: вопрос касается
всех людей. Положительная разность разностей сама по себе не доказывает улучшение
Mary: смотрите оба составляющих эффекта. Нулевой эффект внутри интервала не
доказывает отсутствие эффекта. Интервалы предварительные: всего четыре test-группы,
без поправки на множественные сравнения и без учёта неопределённости подбора probes.

## 7. Сопоставить с full-context QA

Готовые QA-результаты Qwen3-4B находятся в родительской папке:

```text
../artifacts/full-context-canonical-qwen3-4b/report.json
../artifacts/full-context-shuffled-qwen3-4b/report.json
```

Здесь пути указаны относительно `probe_training`. Описание результатов —
[RESULTS.md](../RESULTS.md). Сопоставляйте одинаковые split, инструкции, порядок
фактов и метрики. Не отбирайте для probes только примеры, где QA верен.
QA заново запускать для обучения probes не требуется.

Эти probes проверяют линейную декодируемость относительного порядка людей.
Они не доказывают causal use, полное восстановление состояния или возможность
работы LLM после удаления исходного текста. Три validation-группы также
ограничивают надёжность индивидуального подбора параметров.

## Повторный запуск и ошибки

- Скрипты отказываются перезаписывать output-папку. Для нового запуска задайте
  новый путь, например `probes-qwen3-4b-rerun-01`.
- После прерывания обучения могут остаться отдельные checkpoints. Возобновления
  с них нет: используйте новую output-папку. Итоговый `report.json` записывается
  только после завершения обучения и сравнения условий.
- При несовпадении хешей или ID проверьте соответствие базы и feature manifest;
  после изменения базы признаки необходимо извлечь заново.
- При CUDA OOM уменьшите batch size извлечения признаков. Это не меняет сетку
  параметров probes.
- Не смешивайте partial/smoke features, признаки разных моделей или ревизий.
