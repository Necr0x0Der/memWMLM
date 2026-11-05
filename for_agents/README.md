# For agents: working conventions

This folder is the shared onboarding and workflow layer for coding agents working
in this repository. Read it before starting an experiment.

## Start here

- **Project big idea:** [`big-idea.md`](big-idea.md)
- **Repo conventions:** [`repo-conventions.md`](repo-conventions.md)
- **Handoff template:** [`handoff.md`](handoff.md)
- **GitHub issue template:** [`.github/ISSUE_TEMPLATE/experiment.md`](../.github/ISSUE_TEMPLATE/experiment.md)
- **OpenCode skill:** [`.opencode/skills/github-issues/SKILL.md`](../.opencode/skills/github-issues/SKILL.md)

The big idea is informal context, not a settled claim. It should be updated when
experiments change the picture.

## Core workflow

1. **Read the relevant issue / decision** before starting work.
2. **Create or update one GitHub issue per experiment.** It must state:
   - claim;
   - experiment and protocol;
   - data/config;
   - controls and baselines;
   - expected result / falsification;
   - artifacts;
   - result and caveats.
3. **Run the experiment** and save artifacts under `runs/`, `configs/`,
   `datasets/`, or task-specific folders, as the existing repo does.
4. **Record results in the issue comments**, not only in commit messages.
5. **Link the issue from `docs/decisions.md`** or a relevant README.
6. **Write a handoff note** for the next agent: what was done, what is open, which
   files/checkpoints matter.
7. **Close the issue** only after results and caveats are recorded.

## Git workflow

- Work on a dedicated branch.
- Push the branch to `origin` and open a pull request.
- **Do not push directly to `main`** unless the maintainer explicitly authorizes it.
- The PR description must link the issue and state the scope of the change.
- Do not force-push shared branches.

## Change safety and ownership

- Do not break existing functionality unless the issue/PR explicitly says so and
  documents the migration.
- Preserve old configs, checkpoints, CLI behavior, and historical results unless a
  deliberate change is declared and tested.
- If a change touches another author's/agent's code, ask that owner for approval
  and record the approval in the issue or PR.
- Keep changes minimal; do not rewrite unrelated code.

## Research results for later stages

- Issue comments must contain exact metrics, controls, artifact paths, and caveats.
- Add a dated entry to `docs/decisions.md` and link the issue.
- State explicitly what later stages may rely on and what remains unproven.
- Do not silently reinterpret or rename historical results.

## Repo conventions

- Keep **data**, **memory architecture**, and **experiment protocol** separate.
- Every experiment declares its claim and controls explicitly.
- Report train/test, matched exposure budgets, parameter counts, seeds, and
  provenance.
- Do not silently rename or reinterpret historical results; record prompt and
  dataset provenance.
- Treat `shuffled_memory` / `null_memory` and novel-instance splits as mandatory
  causal controls where applicable.
- Do not claim general semantic memory from task-specific gains.
- Do not commit secrets, tokens, or credentials.
- Prefer a `GITHUB_TOKEN` environment variable or credential helper over a token
  embedded in `git remote` URLs.

## Issues vs artifacts

GitHub issues are for the **experiment record**; repository files are for the
**reproducible artifacts**. An issue should point to the files and checkpoints,
and a file/result should point back to the issue number.

Useful labels:

- `experiment`
- `result`
- `architecture`
- `data`
- `blocked`
- `open-question`

## Notes for other agents

Read the big idea first, then the specific issue, then `docs/decisions.md`.
Do not start from a later architecture assumption if the issue records an open
claim. The current repository state is the source of truth.
