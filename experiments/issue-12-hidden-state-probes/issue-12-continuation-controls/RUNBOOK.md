# Запуск на машине с GPU

Все команды выполняются из корня репозитория. Требуются существующий `.venv`
с PyTorch/Transformers и датасет `experiments/issue-12-hidden-state-probes/artifacts/db`
с `manifest.json`, `train.jsonl`, `validation.jsonl`, `test.jsonl`.
Датасет находится в игнорируемом artifacts: при переносе репозитория его нужно
скопировать отдельно. Старые feature caches для этого эксперимента не нужны.

## 1. Проверка

```bash
continuation_root='experiments/issue-12-hidden-state-probes/issue-12-continuation-controls'
continuation_db='experiments/issue-12-hidden-state-probes/artifacts/db'

.venv/bin/python "$continuation_root/probe_controls.py" \
  --dataset-dir "$continuation_db" --audit-only

.venv/bin/python -m pytest -q "$continuation_root/test_continuation_controls.py"

.venv/bin/python -c 'import torch; print("CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available()'
```

Аудит должен показать 408/72/96 строк. Тесты используют маленькую случайную
Qwen: это проверка кода, не результат исследования.

Extractor загружает pinned модель только из локального HF-cache. Если её ещё
нет, отдельно скачать на GPU-машине:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-4B',
    revision='1cfa9a7208912126459214e8b04321603b3df60c',
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'],
)
PY
```

## 2. Извлечение признаков на GPU

```bash
mkdir -p "$continuation_root/artifacts/logs"
set -o pipefail
.venv/bin/python -u "$continuation_root/extract_controls.py" \
  --dataset-dir "$continuation_db" \
  --output-dir "$continuation_root/artifacts/features-qwen3-4b" \
  --instruction none --fact-order shuffled \
  --layers 8,16,28,36 --device cuda --dtype bfloat16 \
  --batch-size 4 --max-length 512 \
  2>&1 | tee "$continuation_root/artifacts/logs/extract.log"
```

При OOM уменьшить batch size до 1 и указать **новую** output directory;
существующие каталоги намеренно не перезаписываются. Автоматического resume
нет. Обрезание текста запрещено. Ошибка длины продолжений требует проверки
токенизатора/протокола до запуска, а не отключения проверки.

Успешное окончание создаёт `features-qwen3-4b/manifest.json`. В manifest:
точные продолжения, аудит длины, dataset hashes, hashes признаков и prompts,
revision модели, версии библиотек. Незавершённый cache без manifest не годится
для обучения проб.

## 3. Обучение проб и сравнения на CPU

```bash
.venv/bin/python -u "$continuation_root/probe_controls.py" \
  --dataset-dir "$continuation_db" \
  --feature-dir "$continuation_root/artifacts/features-qwen3-4b" \
  --output-dir "$continuation_root/artifacts/probes-qwen3-4b" \
  --instruction none --fact-order shuffled \
  --poolings last_token,mean_pool --l2-values 0.0001,0.001 \
  --max-iter 100 --device cpu --threads 1 \
  --seed 12012 --bootstrap-samples 2000 \
  2>&1 | tee "$continuation_root/artifacts/logs/probes.log"
```

GPU на этом шаге не нужна. Сначала выбираются все 70 проб на train/validation,
затем открывается test. Не подбирать параметры по итоговому test.

## Что смотреть и передать для обсуждения

В `artifacts/probes-qwen3-4b/`:

- `SUMMARY.md` — готовые таблицы метрик и парных разниц;
- `report.json` — полный отчёт, конфигурации, метрики по каждой цели, интервалы;
- `test_predictions.jsonl` — 960 строк (10 условий × 96 ситуаций);
- `selection.json` — решения по validation, записанные до открытия test;
- 140 `.pt` — 70 основных и 70 shuffled-label классификаторов.

В первую очередь читать `questions_vs_neutrals` и `general_vs_neutrals`, затем
их сравнения с каждым нейтральным контролем отдельно. Проверить, улучшились
ли нейтральные условия относительно `no_continuation`. Не делать вывод о
равенстве условий только потому, что интервал включает ноль.

Для разбора результатов достаточно сначала передать `SUMMARY.md`,
`report.json`, `test_predictions.jsonl` и feature `manifest.json`. Сохранить
также полный cache, checkpoints и логи для воспроизводимости.
