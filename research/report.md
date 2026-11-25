# AIR-005 — Transformer From Scratch: experimental report

Educational forward-pass reproduction; synthetic float64 inputs, three fixed seeds (11, 22, 33). See protocol and limitations.

## Complete results

| Experiment | Configuration | Metric | Measured mean ± SD | Reference mean ± SD | Reference type |
|---|---|---|---:|---:|---|
| EXP-001 | unmasked | max absolute output discrepancy | 4.81097e-16 ± 1.69589e-16 | 0 ± 0 | PyTorch agreement ideal |
| EXP-001 | causal | max absolute output discrepancy | 2.96059e-16 ± 1.28198e-16 | 0 ± 0 | PyTorch agreement ideal |
| EXP-002 | heads=1 | max absolute output/weight discrepancy | 2.82182e-16 ± 6.25784e-17 | 0 ± 0 | PyTorch agreement ideal |
| EXP-002 | heads=2 | max absolute output/weight discrepancy | 4.57967e-16 ± 2.01587e-16 | 0 ± 0 | PyTorch agreement ideal |
| EXP-002 | heads=4 | max absolute output/weight discrepancy | 4.81097e-16 ± 2.56395e-16 | 0 ± 0 | PyTorch agreement ideal |
| EXP-003 | unmasked | max absolute output discrepancy | 6.4763e-16 ± 1.397e-16 | 0 ± 0 | PyTorch agreement ideal |
| EXP-003 | causal | max absolute output discrepancy | 5.36608e-16 ± 1.15556e-16 | 0 ± 0 | PyTorch agreement ideal |
| EXP-004 | unmasked | max prefix output change | 2.25223 ± 0.892317 | 0 ± 0 | unchanged prefix ideal |
| EXP-004 | causal | max prefix output change | 0 ± 0 | 0 ± 0 | unchanged prefix ideal |
| EXP-005 | padded + fully masked | max absolute output discrepancy | 2.59052e-16 ± 1.69589e-16 | 0 ± 0 | PyTorch agreement ideal |
| EXP-005 | masked keys | maximum masked weight | 0 ± 0 | 0 ± 0 | zero masked mass ideal |
| EXP-006 | without positions | max permutation discrepancy | 5.18104e-16 ± 1.28198e-16 | 0 ± 0 | permutation equivariance ideal |
| EXP-006 | with positions | max permutation discrepancy | 1.16606 ± 0.475627 | 0 ± 0 | permutation equivariance ideal |
| EXP-007 | scale=1 | max absolute output discrepancy | 2.22045e-16 ± 1.11022e-16 | 0 ± 0 | PyTorch agreement ideal |
| EXP-007 | scale=10 | max absolute output discrepancy | 2.03541e-15 ± 1.64797e-15 | 0 ± 0 | PyTorch agreement ideal |
| EXP-007 | scale=100 | max absolute output discrepancy | 0 ± 0 | 0 ± 0 | PyTorch agreement ideal |
| EXP-007 | scale=1000 | max absolute output discrepancy | 0 ± 0 | 0 ± 0 | PyTorch agreement ideal |
| EXP-008 | T=8 | attention weight elements | 128 ± 0 | 128 ± 0 | 2 × T² analytic count |
| EXP-008 | T=16 | attention weight elements | 512 ± 0 | 512 ± 0 | 2 × T² analytic count |
| EXP-008 | T=32 | attention weight elements | 2048 ± 0 | 2048 ± 0 | 2 × T² analytic count |
| EXP-008 | T=64 | attention weight elements | 8192 ± 0 | 8192 ± 0 | 2 × T² analytic count |

## Interpretation

Unmasked and causal attention, multi-head attention and complete blocks agreed with matched PyTorch references near float64 precision in the tested settings. Perturbing future tokens changed unmasked prefix outputs (mean maximum change 2.25223) but left causal prefixes exactly unchanged. Masked weights were exactly zero, including fully masked rows. Without positions, unmasked block outputs were permutation equivariant up to rounding; fixed positions changed this behavior. Q/K scales up to 1000 remained finite and matched reference outputs. Doubling sequence length quadrupled dense attention weight count. No model training or language quality claim.

## Verification

34 tests passed; 8 executed notebooks; all 63 records reproduced within absolute tolerance 1e-9. Hosted CI pending.
