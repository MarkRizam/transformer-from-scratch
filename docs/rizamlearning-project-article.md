# AIR-005 — Transformer From Scratch

Research question: how accurately can transparent transformer components reproduce reference forward computations, and how do masking, positional information and sequence length affect output behavior?

Implementations use NumPy, with PyTorch serving as a separate reference. Controlled experiments demonstrate attention agreement, causal prefix invariance, key masking, permutation equivariance and quadratic dense-attention storage. See the repository's research report and raw benchmark data. The forward-only scope does not support training or generation. Publication into the learning website should use an immutable verified commit after audit and hosted CI.
