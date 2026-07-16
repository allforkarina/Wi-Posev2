# Final Report Evidence

`results/final_report_seed42_v6/` contains the final single-seed report run
completed on 2026-07-05. It preserves the experiment registry, deterministic
manifests, training curves, test-set overall/action/environment/joint metrics,
and runtime measurements. It is evidence for reproducibility, not a
multi-seed statistical claim.

| IDs | Matched factor | Control |
| --- | --- | --- |
| A1-A4 | Axial attention order | Same split, seed, joint decoder, and bone loss. |
| D1/D3 | Decoder structure | Compare MLP and hierarchical decoders with A1. |
| B1 | Bone-loss prior | Compare zero bone-loss weight with A1. |
| F1-F5 | Few-shot adaptation depth | Fixed 540-shot target protocol; change only trainable group. |
| V2-V4 | Few-shot scale | Use the selected group and vary target examples. |
| Both split modes | Split robustness | Repeat the same matrix for random-frame and temporal-block16. |

Interpret results with the linked metric rows: MPJPE and PCK measure overall
accuracy; per-action and per-joint rows diagnose motion and anatomical error;
runtime metrics describe the cost of the evaluated checkpoint.
