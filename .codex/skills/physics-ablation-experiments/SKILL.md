---
name: physics-ablation-experiments
description: Use this skill whenever the user wants to plan, design, implement, analyze, or write up ablation experiments for Wi_Posev2 or similar CSI-to-pose models. This includes requests about removing or adding attention, axial modes, decoders, loss terms, skeleton priors, feature visualization, cross-domain finetuning variants, or explaining why a module improves performance. The skill forces every ablation to connect a physical or structural hypothesis to a matched control, metrics, feature evidence, and a defensible interpretation.
---

# Physics-Driven Ablation Experiments

Use this skill to turn broad ablation ideas into a rigorous experiment plan for the Wi_Posev2 project. The goal is not just to compare numbers, but to connect each module to a physical or structural claim about CSI sensing and human pose regression.

## Core Principle

Every ablation item needs four linked parts:

1. **Physical or structural hypothesis**: what property of CSI, motion, antenna response, subcarrier response, temporal dynamics, or skeleton topology the module is expected to exploit.
2. **Matched comparison**: the smallest controlled change that isolates the module while keeping data split, training schedule, model size, losses, seeds, and evaluation protocol as comparable as practical.
3. **Performance evidence**: metrics that show whether the change improves or degrades pose regression.
4. **Mechanism evidence**: feature visualizations or error breakdowns that explain why the performance changed.

If one of these parts is missing, state the gap before proposing code changes.

## First Response Pattern

When the user asks for ablation work, first restate the experimental objective in this form:

```text
Goal: [what claim the ablation should test]
Assumption: [what is already implemented or available]
Success criteria: [metric change + mechanism evidence + reproducible command/log]
Smallest next step: [one concrete experiment or implementation change]
```

Then create a compact plan. Do not jump straight to modifying model code unless the target module and comparison are clear.

## Ablation Matrix

Build an ablation matrix before implementation. Use this template:

| ID | Module / factor | Physical hypothesis | Baseline | Variant | Controlled settings | Metrics | Mechanism evidence | Expected failure mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | [module] | [CSI or pose reason] | [control] | [change] | [same data, seed, epochs, loss, etc.] | [MPJPE, PCK, bone error] | [feature viz or error slice] | [what should get worse] |

Prefer small, interpretable factors:

- **Antenna mixing / channel response**: tests whether antenna channels encode direction and spatial phase-difference cues.
- **Subcarrier-frequency processing**: tests whether frequency response patterns help localize body structure.
- **Temporal processing**: tests whether the 64-step motion signal carries Doppler-like movement cues.
- **Axial attention order**: tests whether spatial-first or temporal-first selection better handles low-SNR CSI.
- **Joint cross-attention decoder**: tests whether joint queries can extract pose-specific features from global CSI embeddings.
- **Hierarchical decoder**: tests whether skeleton topology and parent-child joint dependencies improve structured pose regression.
- **Bone or topology-aware losses**: tests whether skeleton constraints reduce physically implausible joint layouts.
- **Few-shot freeze tiers**: tests whether domain adaptation benefits from preserving low-level CSI features while adapting pose-specific layers.

## Experimental Controls

Keep the comparison clean:

- Use the same dataset root, source/target environment filters, train/val/test splits, few-shot indices, seed, batch size, epochs, optimizer, scheduler, and losses unless the ablation explicitly changes one of them.
- Store every changed architecture choice in checkpoint `train_config` so `eval.py` rebuilds the correct model.
- Use separate output directories with names that encode the factor, for example `outputs/ablations/axial_spatial_then_temporal_seed42`.
- Prefer at least three seeds for final claims. A single seed is acceptable only for quick screening and must be labeled as exploratory.
- Do not compare a fully tuned variant against an untuned baseline unless the purpose is clearly an end-to-end system comparison.

## Metrics To Report

Use the existing metrics first so results are comparable with current training and evaluation:

- Validation/test MPJPE.
- PCK at the configured threshold, especially `pck_0_2` when available.
- Coordinate L1 and bone L1 when training logs expose them.
- Per-action, per-environment, and per-joint error when the claim concerns motion, domain transfer, or skeleton structure.

For final tables, report `mean +/- std` across seeds when possible. Include the absolute value and the delta versus the matched baseline.

## Mechanism Evidence

Use the project evaluation tools to explain why a module helped or hurt:

- `eval.py --feature-viz` for research-grade feature visualizations.
- `evaluation/hooks.py` to extract intermediate WiFlow features without changing forward logic.
- `evaluation/feature_viz.py` for antenna response, resblock PCA trajectory, axial attention maps, joint query t-SNE, and feature-pose correlation.
- `evaluation/pose_viz.py` for per-action pose scatter and skeleton comparisons.

Choose evidence that matches the hypothesis:

- Attention claim: show whether attention maps or query embeddings become more joint/action-specific.
- Antenna claim: show channel response differences and whether removing antenna-aware processing worsens left/right or orientation-sensitive joints.
- Temporal claim: show per-action degradation for motion-heavy actions and temporal feature changes.
- Skeleton prior claim: show reduced bone length error, fewer implausible limb layouts, and per-limb improvements.
- Domain adaptation claim: show target-environment improvement while excluding few-shot training frames from evaluation.

## Write-Up Structure

When summarizing ablation results, use this structure:

```text
Hypothesis:
[The physical or structural reason this module should help.]

Comparison:
[Baseline, variant, and controlled settings.]

Result:
[Main metrics and deltas. State whether this is exploratory or multi-seed.]

Mechanism:
[Feature visualization or error breakdown that supports or weakens the hypothesis.]

Interpretation:
[Why the module improves or hurts performance. Mention confounders honestly.]
```

Avoid unsupported claims such as "attention improves performance because it focuses on targets" unless the experiment includes mechanism evidence that aligns with that statement.

## Implementation Guidance

Before code changes:

- Read the relevant model, training, evaluation, and test files.
- Prefer existing CLI switches such as `--axial-mode`, `--decoder-type`, `--mode`, `--eval-envs`, and `--exclude-indices` before adding new arguments.
- If a new ablation switch is needed, keep it narrow, store it in `train_config`, and add a focused test for the shape or configuration contract.
- Do not refactor unrelated model code while adding an ablation.
- Do not commit generated datasets, checkpoints, or large outputs.

Before claiming completion:

- Run the smallest relevant verification command, usually `pytest` for code changes or a short sanity command for training/evaluation wiring.
- If project code or tests are run, activate the `WiFiPose` Conda environment first.
- Report commands run, files changed, and whether the result is a plan, an exploratory run, or final evidence.

## Example Ablation Framing

```text
ID: A1
Module / factor: Joint cross-attention decoder
Physical hypothesis: OpenPose18 joints need different CSI feature selections because torso, limb, and extremity coordinates are implied by different combinations of antenna, subcarrier, and temporal cues.
Baseline: Same encoder with decoder attention disabled or replaced by a matched MLP/head.
Variant: Existing joint cross-attention decoder.
Controlled settings: same split, seed, epochs, loss, optimizer, scheduler, batch size.
Metrics: test MPJPE, PCK_0_2, per-joint MPJPE, bone L1.
Mechanism evidence: joint query t-SNE, attention maps, per-joint error deltas.
Expected failure mode: without joint-specific selection, small or distal joints degrade more than torso joints.
```
