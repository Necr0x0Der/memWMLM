# Запуск source-free memory на Qwen3-4B

Все команды выполнять из корня `memWMLM`, в Bash. Протокол и ограничения:
[README.md](README.md). Статус предыдущей попытки: [RESULTS.md](RESULTS.md).
На этом этапе обучаем адаптер и смотрим тестовые ответы; подмены памяти нет.

## 1. Использовать уже извлечённые признаки

**Повторное извлечение не нужно**, если сохранились признаки предыдущего
эксперимента Qwen3-4B. Обучение soft prefix использует те же situation-only
векторы, что probes; менять формат признаков не требуется.

В прежних runbook указаны два подходящих источника:

- `experiments/issue-12-hidden-state-probes/probe_training/artifacts/features-qwen3-4b`
- `experiments/issue-12-hidden-state-probes/probe_training/artifacts/token-features-qwen3-4b`

Второй cache тоже подходит: `--save-token-states` добавляет отдельные токены,
но сохраняет и три pooled-представления. Индивидуальные токены здесь не нужны.
Наличие больших token-state tensors может замедлить чтение shards.

Укажите **реальный путь на машине, где лежат признаки**:

```bash
source_free_features='experiments/issue-12-hidden-state-probes/probe_training/artifacts/features-qwen3-4b'
source_free_dataset='experiments/issue-12-hidden-state-probes/artifacts/db'
source_free_output='experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/pilot-none-shuffled'
```

Нужны `manifest.json` и все перечисленные в нём shards для `train`,
`validation`, `test`. Один manifest, report или checkpoint обученного probe
не заменяет feature-cache. Признаки Qwen3-1.7B для этого запуска не подходят.

Проверьте manifest:

```bash
.venv/bin/python - "$source_free_features" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
m = json.loads((p / 'manifest.json').read_text())
assert m['model']['name'] == 'Qwen/Qwen3-4B'
assert m['model']['requested_revision'] == '1cfa9a7208912126459214e8b04321603b3df60c'
assert 28 in m['features']['layer_indices']
assert not m['tokenization']['chat_template_applied']
assert m.get('max_examples_per_split') is None
for split in ('train', 'validation', 'test'):
    shards = m['splits'][split]['shards']
    assert shards, split
    for shard in shards:
        assert (p / split / shard['path']).is_file(), shard['path']
    print(split, 'shards:', len(shards))
print('model:', m['model']['name'])
print('layers:', m['features']['layer_indices'])
print('pooling:', m['features']['pooling'])
PY
```

Это быстрая проверка наличия файлов. Хеши shards, соответствие dataset и ID
проверит `run.py` при загрузке. Для пилота нужны `mean_pool`, `last_token`,
`sentence_end_mean`; дополнительные слои в cache допустимы.

## 2. Проверить GPU, модель и реализацию

```bash
nvidia-smi
.venv/bin/python -c 'import torch, transformers; print(torch.__version__, transformers.__version__); print("CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available()'
.venv/bin/python -m pytest -q experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/test_source_free.py
```

Для ответного прохода всё равно нужны **веса Qwen3-4B**: cache признаков
заменяет только повторное чтение ситуаций, а не саму отвечающую модель.
Скрипт загружает pinned revision локально через Hugging Face cache и не скачивает
веса автоматически. Используйте тот же HF-cache, что в предыдущих запусках.
Если нужна явная загрузка на новую машину с интернетом:

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

Нужна VRAM и для градиентов по входу: хотя веса Qwen заморожены, backward
проходит через неё. Минимальный объём VRAM для этого запуска ещё не измерен.

## 3. Только если feature-cache утрачен: извлечь заново

При наличии совместимого cache **пропустите этот шаг**. Старые признаки не
перезаписывать; output-dir extractor должен быть новым.

```bash
source_free_features='experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/features-qwen3-4b'
.venv/bin/python experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir "$source_free_dataset" \
  --output-dir "$source_free_features" \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --layers 28 --device cuda --model-dtype bfloat16 \
  --output-dtype float16 --batch-size 4
```

Extractor сохраняет все три pooling. `--save-token-states` не нужен.
Не использовать `--max-examples-per-split`: это даст неполный cache для smoke test,
который основной эксперимент намеренно отклоняет.

## 4. Обучить и получить ответы на test

Зафиксированный пилот: `none/shuffled`, слой 28, три pooling, четыре memory-слота,
пять эпох и seed 12012. По validation NLL выбираются pooling и эпоха;
после выбора запускается генерация на test, сначала с памятью, потом без неё.
Test-признаки не участвуют в выборе адаптера.

```bash
mkdir -p experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/logs
set -o pipefail
.venv/bin/python -u experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/run.py \
  --dataset-dir "$source_free_dataset" \
  --feature-dir "$source_free_features" \
  --output-dir "$source_free_output" \
  --instruction none --fact-order shuffled \
  --poolings mean_pool,last_token,sentence_end_mean --layers 28 \
  --slots 4 --epochs 5 --batch-size 2 --lr 1e-4 --seed 12012 \
  --device cuda --dtype bfloat16 --max-new-tokens 64 \
  2>&1 | tee experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/logs/pilot-none-shuffled.log
```

`source_free_output` должен отсутствовать до запуска; родительская папка может
существовать. Для повторной попытки выбрать новое имя output-dir и log.
Скрипт сообщает train/validation NLL после каждой эпохи и прогресс каждых
50 тестовых ответов. Ожидаются 15 записей эпох (3 pooling × 5), 672 ответа
с памятью и 672 без памяти. Итоговый `report.json` появляется после обоих режимов.

## 5. Посмотреть метрики и сами ответы

```bash
.venv/bin/python - "$source_free_output" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
r = json.loads((p / 'report.json').read_text())
print('Selected:', r['selected'])
for mode, m in r['test'].items():
    print(mode,
          'macro:', round(100 * m['mean_macro_accuracy'], 2),
          'joint:', round(100 * m['joint_exact_targets_accuracy'], 2),
          'invalid:', m['invalid_answers'], '/', m['answers'])
for mode in ('memory', 'question_only'):
    print('\nExamples:', mode)
    with (p / f'test-{mode}.jsonl').open() as f:
        for _, line in zip(range(5), f):
            row = json.loads(line)
            print(row['id'], row['question'])
            print('Target:', row['target'], 'Prediction:', row['prediction'])
            print('Raw:', repr(row['response']))
PY
```

`macro` — средняя balanced accuracy семи вопросов; `joint` — все семь ответов
правильны в одной ситуации. Invalid считается ошибкой. Оцениваются 96 ситуаций,
но независимых abstract-layout групп всего четыре.

Сохранить вместе: `protocol.json`, `history.json`, `selected.pt`, оба JSONL
с ответами, `report.json`, лог и manifest/features. Сначала проверить invalids и
сырые ответы. Разница с question-only — предварительный сигнал пригодности
интерфейса; без подмен и oracle причинный вывод не делаем. Старое full-context QA
имело другой generation budget, поэтому сравнение с ним не является matched.

## 6. Если запуск остановился

- **CUDA unavailable:** запускать на GPU-машине; CPU-тесты не заменяют обучение 4B.
- **Модель не найдена локально:** проверить HF-cache и pinned revision; выполнить
  явную загрузку из шага 2 при необходимости.
- **Feature/dataset mismatch или missing shard:** перенести полный cache и его
  исходный dataset. Не редактировать хеши manifest для обхода проверки.
- **Output directory exists:** оставить предыдущие артефакты и выбрать новое имя.
- **CUDA out of memory:** уменьшить batch size до 1 и начать отдельный запуск.
  Это изменение конфигурации; записать его. Число optimizer steps изменится.
- **Нет report.json:** запуск не завершился. История, лучший checkpoint и частичные
  JSONL могут сохраниться; не считать их полным test-результатом.
- **Много invalid/обрезанных ответов:** сохранить текущую оценку; увеличение
  generation budget оформить отдельным запуском, а не незаметной заменой результата.

Автоматического resume и отдельного CLI для eval-only пока нет. Подмена памяти,
oracle, новые инструкции и расширение сетки слоёв — следующие эксперименты,
не часть этого пилота.
