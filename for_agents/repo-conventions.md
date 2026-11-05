# Repository conventions

## 1. Experiment record

Every experiment gets one GitHub issue. The issue should contain:

- **Claim:** one falsifiable statement, one claim at a time.
- **Experiment:** task, protocol, model, memory variant.
- **Data/config:** dataset, split, prompt version, checkpoint, seed.
- **Controls:** baselines, `null_memory`, `shuffled_memory`, novel-instance,
  counterfactual source swap, matched exposure budget.
- **Expected result / falsification:** what would confirm or refute the claim.
- **Artifacts:** paths under `runs/`, `configs/`, `datasets/`, `RESULTS.md`.
- **Result:** metrics and qualitative failures.
- **Caveats:** confounders, distribution shift, overfitting, unknown provenance.

## 2. Decisions

`docs/decisions.md` is append-only. New results should be added as dated entries,
not by rewriting old measurements. Link the relevant issue number when possible.

## 3. Artifacts

Save:

- checkpoint paths and step numbers;
- exact configs and seeds;
- dataset hashes / manifests;
- train/test metrics and control metrics;
- error examples, especially shuffled-memory failures;
- run logs where they are useful.

Do not store per-example runtime memory state as learned global knowledge.

## 4. Claims and controls

- Separate **storage**, **retrieval**, and **task adaptation**.
- Do not infer general memory from a task-specific gain.
- Do not infer model-size effects from a single task.
- Keep class A (data + light reasoning) and class B (structural computation)
  separate.
- Frozen backbone is an experimental constraint; report adaptation variants
  separately.

## 5. Git and secrets

- Commit messages should reference the issue or experiment ID.
- PR descriptions should link the issue.
- Do not commit tokens, API keys, or credentials.
- Do not print tokens in logs or responses.
- Prefer environment variables or credential helpers over tokens embedded in
  remote URLs.

## 6. Handoff

At the end of an experiment, leave a short handoff note in the issue or in
`for_agents/handoff.md`:

- current state;
- files/checkpoints to inspect;
- what remains open;
- what the next agent should not assume.
