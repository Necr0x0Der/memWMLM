---
name: GitHub Issues Workflow
description: Use GitHub issues to record experiments, controls, artifacts, results, and handoffs for this repository.
---

## When to use this skill

Use it whenever you start, run, or finish an experiment in this repository. The
goal is to leave a durable record that other agents can read.

## Read first

From the repository root:

1. `for_agents/README.md`
2. `for_agents/big-idea.en.md` or `for_agents/big-idea.ru.md`
3. `docs/decisions.md`
4. the relevant GitHub issue

GitHub's `/issues` endpoint also returns pull requests. Do not confuse them with
issue records unless the PR is the experiment record.

## Workflow

1. Find the issue for the experiment.
2. Create an issue if none exists, using `.github/ISSUE_TEMPLATE/experiment.md`.
3. Include:
   - claim;
   - strategic context: why it matters for the project big idea;
   - protocol and model;
   - data/config and seed;
   - controls and baselines;
   - expected result / falsification;
   - artifact paths;
   - compatibility and ownership approvals;
   - result and caveats.
4. Save artifacts under `runs/`, `configs/`, `datasets/`, or task-specific paths.
5. Put results and caveats in issue comments, not only in commit messages.
6. Link the issue from `docs/decisions.md` or the relevant README.
7. Write a handoff note for the next agent.
8. Close the issue only after result and caveats are recorded.
9. Work on a branch and open a PR; do not push directly to `main` unless the
   maintainer explicitly authorizes it.
10. Do not break existing functionality unless the change is explicitly scoped,
    documented, and tested.
11. If you touch another author's/agent's code, ask that owner and record approval.

## GitHub API examples

Use a token from an environment variable or credential helper. Do not print or
commit tokens.

```bash
# list issues and PRs
curl -sS -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/Necr0x0Der/memWMLM/issues?state=all&per_page=50"

# read one issue / PR
curl -sS -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/Necr0x0Der/memWMLM/issues/3"

# create an issue
curl -sS -X POST -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/Necr0x0Der/memWMLM/issues" \
  -d '{"title":"[experiment] ...","body":"...","labels":["experiment"]}'

# comment
curl -sS -X POST -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/Necr0x0Der/memWMLM/issues/3/comments" \
  -d '{"body":"Result: ..."}'
```

## Rules

- Do not commit secrets.
- Do not claim general memory from a task-specific gain.
- Do not compare metrics without matching target length, exposure budget, prompt
  provenance, and controls.
- Frozen backbone is an experimental constraint, not a final requirement.
- Keep storage, retrieval, and task adaptation separate.
