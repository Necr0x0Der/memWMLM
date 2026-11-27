# Source-free soft-prefix memory: первый тест без исходного описания

Продолжение [issue #12](https://github.com/Necr0x0Der/memWMLM/issues/12)
и [PR #14](https://github.com/Necr0x0Der/memWMLM/pull/14).
Отдельный протокол: прежние эксперименты и артефакты не меняются.

Пошаговый запуск с уже извлечёнными признаками: [RUNBOOK.md](RUNBOOK.md).

## Что проверяем

Можно ли обучить affine-адаптер, который превращает один сохранённый вектор
ситуации в четыре виртуальных входных токена, позволяющих **замороженной
Qwen3-4B отвечать на новые тестовые ситуации без исходного описания**?

```
ситуация без вопроса → frozen Qwen → h слоя L и выбранного pooling
                                           ↓
                                  train-standardization
                                           ↓
                               обучаемый Linear(H, 4*D)
                                           ↓
                           [4 soft tokens] [chat-вопрос]
                                           ↓
                                  frozen Qwen → ответ
```

Исходный текст и его KV-cache никогда не передаются в ответный проход.
Виртуальные токены передаются через `inputs_embeds`, перед полным chat-prompt.
Вопрос адаптеру недоступен: все семь вопросов об одном мире используют одну
память. Обучение — teacher forcing на `FINAL: <answer>` и EOS, loss только
на ответе. Градиент проходит через Qwen к адаптеру; веса Qwen заморожены.
Признаки и статистики их нормализации не получают градиент; статистики считают
только на train выбранного условия. Инициализация масштаба адаптера использует
стандартное отклонение первых 4096 входных эмбеддингов Qwen.

Это **проверка пригодности представления через обученный интерфейс**, а не
доказательство, что исходная Qwen использовала готовое состояние мира.
Адаптер affine, но не мал по числу весов: при H=D=2560 и K=4 это 26 224 640
параметров. Он может выучить часть задачи. Четыре слота не добавляют информации
к исходному h, но обучают способ её подачи.

## Зафиксированный пилот до просмотра новых test-ответов

- Модель: Qwen/Qwen3-4B, revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Старый dataset `issue-12-hidden-state-probes/artifacts/db`, неизменённые splits.
- Одно условие `none/shuffled`: 408 train, 72 validation, 96 test ситуаций;
  по семь вопросов на ситуацию (2856/504/672 ответов).
- Слой **28** фиксирован для пилота; pooling-кандидаты: `mean_pool`,
  `last_token`, `sentence_end_mean`. Последний — среднее концов шести фактов.
- Один отдельный адаптер на каждый pooling; K=4, 5 эпох, AdamW lr=1e-4,
  weight decay=0, batch size=2, gradient clipping=1, seed=12012.
- Выбираем pooling и эпоху по **минимальной validation answer-token NLL**.
  Равенство разрешается в пользу первого кандидата/ранней эпохи.
  Не выбираем по test; не переобучаем на train+validation.
- После выбора — greedy generation на всех 96 тестовых мирах и 7 вопросах,
  thinking disabled, max_new_tokens=64, без constrained decoding.
- Основная метрика: mean per-question balanced accuracy; дополнительные:
  все семь targets правильны одновременно, per-question accuracy, invalids,
  полные сырые ответы. Это не восстановление полного мира.
- Сравнение: та же замороженная модель, тот же вопрос и prompt, но **без памяти**.
  Этот контроль отличается длиной контекста и не является matched null-prefix.
- Критерий первичного положительного сигнала: качество memory выше question-only
  и положительная разница по совместной точности семи targets. Это описательный
  результат пилота, не статистическое доказательство причинного использования.
- Подмены, null-prefix и oracle **отложены по запросу пользователя**. Отсутствие
  этих контролей запрещает вывод «модель причинно использует состояние».
- Существующий test уже изучался в других экспериментах. Независимых test-групп
  всего четыре. Разброс по seed и доверительные интервалы здесь не оцениваются.

Скрипт поддерживает несколько слоёв, например `--layers 8,16,28,36`, но это
отдельное расширение сетки: зафиксировать его **до просмотра test**. При смене
условий или бюджета использовать новый output-dir, не перезаписывать пилот.
Сравнение с прежними 95% full-context QA не является matched: там был бюджет
2048 токенов. Здесь целевое обучение на короткие ответы и лимит 64. При большом
числе обрезанных/невалидных ответов результат считается ограниченным этим
протоколом; не менять парсер или бюджет молча после test.

## Запуск

Из корня репозитория. Нужны CUDA, локальные pinned-веса Qwen3-4B и достаточно
VRAM для backward через замороженный backbone. Заморозка не отменяет хранения
активаций. Автоматического скачивания весов скрипт не делает.

### 1. Признаки

Если сохранён совместимый полный feature-cache Qwen3-4B с тремя pooling и
слоем 28, использовать его. Старые отчёты/чекпойнты probes признаков не заменяют.
Иначе запустить существующий extractor; он сохраняет все три pooling:

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/features-qwen3-4b \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --layers 28 --device cuda --model-dtype bfloat16 \
  --output-dtype float16 --batch-size 4
```

### 2. Обучение, выбор на validation, тестовые ответы

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/run.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --feature-dir experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/features-qwen3-4b \
  --output-dir experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/pilot-none-shuffled \
  --instruction none --fact-order shuffled \
  --poolings mean_pool,last_token,sentence_end_mean --layers 28 \
  --slots 4 --epochs 5 --batch-size 2 --lr 1e-4 --seed 12012 \
  --device cuda --dtype bfloat16 --max-new-tokens 64
```

Каждый новый запуск требует несуществующий output-dir. Проверяются хеши dataset,
manifest/shards признаков, соответствие модели/revision, совпадение ID и
непересечение abstract-layout групп. Test-признаки загружаются после окончательного
выбора; до этого test-метаданные читаются только для аудита splits.

Выходы:

- `protocol.json`: параметры и provenance до обучения/оценивания;
- `history.json`: train/validation NLL всех кандидатов и эпох;
- `selected.pt`: лучшие веса, train-нормализация, конфигурация и число параметров;
- `test-memory.jsonl`: реальные ответы по памяти;
- `test-question_only.jsonl`: реальные ответы без памяти;
- `report.json`: выбранный кандидат, метрики обоих условий, invalids, хеши.

Вопросы на тесте не содержат ground truth. Правильный ответ сохраняется лишь
для scoring. Старый парсер `FINAL:` используется без изменений; invalid считается
ошибкой. Генерация по одному вопросу упрощает проверку отсутствия cache-переноса,
но не оптимизирована по скорости. Автоматический resume пока не реализован;
при прерывании остаются история, лучший checkpoint и уже записанные ответы.

### Проверки реализации

```bash
.venv/bin/python -m pytest -q experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/test_source_free.py
```

Тесты используют маленькую случайную Qwen: градиент к адаптеру, неизменность
backbone, маски loss/padding, отсутствие текста/ответа в question-prompt,
совпадение generation с ручным greedy и полный train→select→test pipeline.
Эти тесты не являются измерением качества Qwen3-4B.

## Текущий статус и handoff

См. [RESULTS.md](RESULTS.md). Локальная запись протокола подготовлена;
новый GitHub issue/комментарий и публикация не выполнялись. Следующий запуск
должен сохранить этот протокол и использовать GPU с pinned-моделью и признаками.
