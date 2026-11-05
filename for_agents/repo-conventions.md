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

## 5. Git and PR workflow

- Work on a dedicated branch, push the branch, and open a pull request.
- **Do not push directly to `main`** unless the maintainer explicitly authorizes it.
- Commit messages should reference the issue or experiment ID.
- PR descriptions should link the issue and state scope, compatibility, and
  ownership approvals.
- Do not force-push shared branches.
- Do not commit tokens, API keys, or credentials.
- Do not print tokens in logs or responses.
- Prefer environment variables or credential helpers over tokens embedded in
  remote URLs.

## 6. Compatibility and ownership

- Do not break existing functionality unless explicitly scoped and documented.
- Preserve old configs, checkpoints, CLI behavior, prompt versions, and historical
  results unless a deliberate change is declared and tested.
- Add tests or migration notes for behavior changes.
- If a change touches another author's/agent's code, ask that owner for approval
  and record the approval in the issue or PR.
- Keep edits minimal; do not rewrite unrelated code.

## 7. Results for later stages

- Record exact metrics, controls, artifact paths, checkpoint/step, seed, and
  caveats in the issue.
- Add a dated entry to `docs/decisions.md` and link the issue.
- State what later stages may rely on and what remains unproven.
- Keep historical measurements intact; do not silently reinterpret or rename
  them.

## 8. Handoff

At the end of an experiment, leave a short handoff note in the issue or in
`for_agents/handoff.md`:

- current state;
- files/checkpoints to inspect;
- what remains open;
- what the next agent should not assume.
