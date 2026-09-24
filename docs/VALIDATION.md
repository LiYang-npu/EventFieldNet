# Validation of the cleaned release

Passed in the recorded Linux/PyTorch2.1.0 CUDA environment:

1. Four consecutive real training batch64 identities, cropped to first8 for CPU comparison: initial parameter hash, candidate scores, six loss scalars, all gradient hashes and all updated parameter hashes exactly match the frozen V00 reference after each update (seed2041).
2. Four real GPU batch64 scratch optimizer updates with independent E/S/T gradient checks, followed by full1550 validation; smoke checkpoint reconstruction passed. Smoke is not formal model performance.
3. One complete scratch training epoch through the real runner, optimizer/scheduler, probes, checkpoint save and evaluator subprocess completed. Integration-only one-epoch budget; not a replacement for24epoch research results.
4. Frozen V00 seed2041 best checkpoint: full1550 validation exactly reproduces57.16256326550931. Full/E/S/T fixed-checkpoint evaluations all completed with strictFP32/noTF32 receipts and the same checkpoint SHA. The final source revision repeated CPU parity and full-val reproduction successfully.
5. Pinned Moment-DETR standalone scoring command executed on saved complete validation predictions. Source hashes match the pinned official files.
6. All distributed Python files parse; all project source, CLI and regression tests pass Ruff F-series checks. Four entry-point regression tests pass. Private machine-path/address/key-name scan passed. File hash manifest verifies archive integrity.

See validation_receipt.json and FILE_REVIEW.csv. Exact CPU parity on a bounded trajectory does not establish deterministic CUDA training. Full24epoch retraining of the cleaned package, fresh dependency installation in an empty environment, Windows training and every conditional helper branch were not certified. The new test CLI reuses the validated heldout adapter, but the cleaned package did not rerun the55 test predictions; the result table is the already completed frozen-checkpoint evaluation, with that provenance disclosed.

A metadata translation bug in an earlier packaging attempt caused smoke evaluation to reject its new checkpoint. It was corrected, covered by an idempotence regression test, and the full smoke was repeated successfully. Failed development receipts remain outside the release source tree.
