# AIR-005 experimental protocol
Educational forward-pass reproduction, not preregistration or novel research. Fix seeds 11, 22, 33 and configurations before executing. CPU float64. PyTorch is a reference only. Report all configurations; compare fresh and committed records with absolute tolerance 1e-9 and zero relative tolerance. No trained weights or external datasets.

- EXP-001: random Q/K/V, two heads, sequence length 6, head size 4; unmasked and causal attention against PyTorch SDPA; maximum absolute output discrepancy.
- EXP-002: multi-head self-attention, width 8, heads 1/2/4, length 6; match all projection weights in PyTorch MultiheadAttention.
- EXP-003: post-norm transformer block, width 8, heads 2, FFN width 16; unmasked and causal outputs against PyTorch TransformerEncoderLayer.
- EXP-004: perturb final three tokens of a six-token sequence; measure maximum absolute change in first three output tokens, causal vs unmasked. No claim about learned language behavior.
- EXP-005: key padding mask, two batches, one fully masked; compare outputs with PyTorch SDPA, measure masked attention mass; attention output zeros do not imply full block output zeros.
- EXP-006: permute a five-token sequence; compare permutation equivariance of unmasked block with and without fixed sinusoidal positions. Nonzero discrepancy with positions is descriptive, not a measure of prediction quality.
- EXP-007: scale Q/K by 1, 10, 100, 1000; compare stable attention output against PyTorch, including large logits. Not a proof for arbitrary finite inputs.
- EXP-008: sequence lengths 8, 16, 32, 64; exact stored attention weight element count for 2 heads and one batch; compare against 2*T*T. This is not total memory or latency benchmarking.

Summarize means and sample SD across three seeds (not confidence intervals). Zero reference for discrepancy metrics is an agreement ideal, not a competing model. No model training, text generation, tokenizer, cross-attention, backpropagation, GPU or real-world language evaluation.
