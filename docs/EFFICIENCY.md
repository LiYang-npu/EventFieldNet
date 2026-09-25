# Memory and training efficiency

The fixed F05 model now avoids unused work and short-lived dense activations. No extra training option is required. Parameter names, checkpoint structure, full video lengths, candidate grids, FP32, loss weights and optimizer update order are preserved.

## Implementation

1. For more than 128 time steps, non-reentrant activation checkpointing recomputes the shared candidate MLP and inherited E/S readouts during backward. It retains compact inputs instead of multiple candidate-by-hidden tensors. Short QV sequences follow the original computation.
2. The two old masked-video E forwards are skipped when the actual local E objective supersedes them. Necessary wrong-query forwards and every active E/S/T objective remain. Old masked-E diagnostic fields are explicitly marked unavailable; `model.retain_legacy_counterfactual_diagnostics = True` restores them at the old compute cost.
3. Candidate geometry and endpoint targets are reused within the same loss call. Disabled structural diagnostics use bounded, no-gradient chunks.
4. The training loop releases completed batch outputs and its three per-batch model caches after gradients, updates and diagnostics. Gradient checks and metric transfers are grouped to reduce GPU synchronization. Cached unused allocator blocks are released once before the separate evaluator; no per-batch `empty_cache` is used.

## Real-data measurements

NVIDIA H20-3e, PyTorch 2.1.0/CUDA12.1, strict FP32/no TF32. One real warmup update was restored before timing. The table measures the actual training loop, including metric collection, on four repeated longest-video batches; it excludes disk feature reads and separate full-validation evaluation.

| Dataset | Batch / maximum length | Peak allocated before → after | Mean step before → after |
|---|---|---:|---:|
| Charades | 4 / 195 | 15.14 → 4.22 GiB | 0.507 → 0.417 s |
| Tacos | 2 / 702 | 89.31 → 22.86 GiB | 0.880 → 0.688 s |

TACoS also passes four updates with batch 4 and the four longest distinct training videos: peak allocated 45.33 GiB, peak reserved 59.77 GiB. This is a capacity check, not an unchanged-batch numerical comparison or an accuracy claim. Released training batch sizes were not changed.

Eight fixed shuffled real training batches per dataset were also measured. All identities, length summaries, allocated/reserved memory and timing aggregates are in [efficiency.json](../results/efficiency.json). Allocated memory is not the same as `nvidia-smi` process usage: the CUDA allocator may reserve more. These bounded timings do not predict an entire epoch exactly.

The profiling adapter uses the already audited moment-only E target for unrated data. It preserves complete E/S/T losses but is distinct from QV ordinal supervision. These measurements do not turn this fixed QV package into a validated Charades/TACoS accuracy pipeline.

## Numerical checks

- Four consecutive real QV CPU training batches: initialization, scores, all losses, all gradients, updated weights and RNG match the pre-optimization reference exactly.
- Long-sequence CPU field/head tests cover padding, optional branches and repeated retained-graph gradient queries, with exact outputs and gradients.
- Real long-video CUDA runs preserve initialization, batch identities and RNG; all tested gradients pass rtol=1e-4/atol=1e-6. After two updates, one near-zero bias differs by 1.224e-6 and misses that fixed parameter tolerance; the original implementation repeated against itself differs by 2.201e-6. The discrepancy is retained in the audit. No claim of deterministic full GPU training is made.
- The frozen original F05 seed2041 checkpoint reproduces all 1550 QV validation prediction rows exactly: MR 57.69770996391387, HD mAP 44.17, HIT@1 74.0. The test set was not used.
- The original nine release tests pass. Existing five-seed accuracy records are retained; these engineering checks do not constitute five new training runs.
