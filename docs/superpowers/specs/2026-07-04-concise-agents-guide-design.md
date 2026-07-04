# Concise AGENTS.md Contributor Guide Design

## Objective

Create a new root-level `AGENTS.md` titled `Repository Guidelines`. The document will be an English contributor guide for Wi-Posev2 and remain within the requested 200–400-word range.

## Content

The guide will use short Markdown sections covering:

- Project structure: root training/evaluation entry points, `models/`, `data/`, `evaluation/`, `scripts/`, `experiments/`, and `tests/`.
- Development commands: Conda activation, `pytest`, a short source-training command, checkpoint evaluation, and final-report pose export.
- Coding conventions: Python 3.10+, four-space indentation, type hints, `pathlib.Path`, `snake_case`, `PascalCase`, and uppercase constants.
- Testing: pytest naming, synthetic fixtures, shape/split/path coverage, and focused test commands.
- Git workflow: concise imperative commits based on recent history, scoped pull requests, commands run, dataset assumptions, and visual evidence when output changes.
- Security: external datasets, no committed checkpoints or machine-specific paths.
- Agent rules: English repository content, Chinese user communication, use of the `WiFiPose` environment, verification before completion, preservation of unrelated changes, and commit/push to the active `codex/` branch after modifications.

## Constraints

The guide will be concise rather than reproducing the removed long-form domain reference. Commands will use existing repository entry points and avoid private dataset paths except generic examples. No other files will be restored or modified as part of this task.

## Verification

Verify that:

- The first heading is exactly `# Repository Guidelines`.
- The document contains 200–400 words.
- Every referenced path and command exists in the current repository.
- Markdown headings and code fences are balanced.
- Only `AGENTS.md` and the approved design/plan artifacts are staged for this task.
