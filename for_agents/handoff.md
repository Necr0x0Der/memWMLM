# Handoff template

Use this at the end of an experiment or before handing work to another agent.

## Current state

- Branch / commit:
- Issue:
- Checkpoint(s):
- Config(s):
- Dataset(s):

## What was done

- Experiment:
- Controls:
- Metrics:
- Compatibility / old behavior preserved:
- Code ownership / approvals:

## What remains open

- Claims not established:
- Confounders:
- Missing controls:
- Next experiment:

## What later stages may rely on

- Established:
- Unproven:
- Reusable artifacts:
- Decision entry / issue link:

## What the next agent should read

- `for_agents/big-idea.md`
- `docs/decisions.md`
- the experiment issue
- relevant `runs/.../RESULTS.md`

## What the next agent should not assume

- Do not treat task-specific gains as general semantic memory.
- Do not treat frozen backbone as a final architectural requirement.
- Do not compare metrics without matching target length, exposure budget, and
  prompt provenance.

## 2026-11-25 experiment handoff (local only)

- Branch: `multicommit-memory`; no commit/push was made.
- Files: `experiments/multicommit-memory/{generate.py, generate_simple.py, generate_two.py, common.py, train.py, evaluate.py, README.md, RESULTS.md}`; `RESEARCH-JOURNAL.md`; `docs/decisions.md` entry.
- Datasets: `datasets/multicommit-memory/`, `datasets/multicommit-simple/`, `datasets/multicommit-simple3/`, `datasets/multicommit-two/`.
- Checkpoints: `runs/multicommit-memory/simple3.pt` (step 1500), `runs/multicommit-memory/two.pt` (step 1500), `runs/multicommit-memory/two_long.pt` (step 3000), plus older diagnostic checkpoints.
- Established: single-entity, ordered three-commit facts are retrievable from memory; `last_commit_only` fails on the original fact; null/no-source controls are near chance; counterfactual source edits are followed.
- Boundary: the 1500-step two-entity checkpoint is undertrained (15.25%); the 3000-step checkpoint reaches 53.75% memory accuracy versus 81.45% full context, with current/previous still low. The complex three-person task also failed. Do not extend the single-entity claim to multi-entity binding.
- Open: second seed; same-name split; free-form generation; mechanism of multi-entity failure; step-1600 NaN.
- Do not assume the complex multi-person or two-entity failure means the writer cannot store facts, nor that the simple result generalizes to arbitrary entity/commit structure.
- Reusable artifacts are local; no remote issue or PR exists.

