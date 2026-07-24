# FarmShare P4 smoke-test result

Run date: 23 July 2026
Slurm job: `1654870`
Node/GPU: `oat-03` / NVIDIA L40S
Job state: completed, exit code 0
Allocated elapsed time: 1 minute 11 seconds

## Setup

- Random-init `OlmoForCausalLM` branch checkpoint.
- Exact trainable parameters: 162,164,736.
- Architecture: 12 layers, width 768, 12 attention heads, SwiGLU
  intermediate size 2048, untied input/output embeddings.
- Both arms restored the identical seeded checkpoint.
- 4,096 examples per skill; 16,384 per arm.
- 512 optimizer updates per arm, batch size 32.
- Same records and same within-skill order; only cross-skill schedule differed.
- Held-out test: 256 examples per skill, 1,024 total, with zero training-skeleton overlap.

## Result

| Skill | Blocked | Interleaved | Difference |
|---|---:|---:|---:|
| A — rotate left | 0.00% | 81.64% | +81.64 pp |
| B — rotate right | 15.23% | 95.31% | +80.08 pp |
| C — reverse | 0.00% | 27.34% | +27.34 pp |
| D — swap pairs | 94.53% | 42.19% | -52.34 pp |
| **Macro / all items** | **27.44%** | **61.62%** | **+34.18 pp** |

Exact correct counts were 281/1,024 for blocked and 631/1,024 for
interleaved. The base checkpoint scored 0/1,024.

## Interpretation

This is a large positive screening signal for interleaving on this synthetic,
random-init task. The blocked model strongly retained the final `D` block but
scored zero on two earlier skills, consistent with catastrophic forgetting and
recency. The interleaved model retained useful performance on all four skills.

This is one deterministic run with one fixed blocked order. It does not estimate
run-to-run uncertainty, separate interleaving from recency/spacing, or establish
an effect for a pretrained OLMo checkpoint or natural-language data.
