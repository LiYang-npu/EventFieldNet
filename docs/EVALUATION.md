# Evaluation protocols

Scoring uses strict FP32 and disabled TF32, raw top30 candidates, hard temporal NMS at0.5 and AP at top10 predictions. Each seed uses its own best Full validation mAP within24 epochs; ties use the original runner selection order. The reported test checkpoint is fixed before test. No seed is removed and no cross-seed ensemble is used.

The SG-derived training evaluator and Moment-DETR standalone evaluator are not numerically identical. SG MRAveragePrecision limits GT windows to10. Its dataset loader shuffles GT lists exceeding max_windows during target preparation. The standalone official evaluator uses all GT windows. Test has12 queries exceeding10 GT windows; official outputs also round to two decimals. An initial rounding-only comparison failed and was retained in the external audit. CPU reconstruction of the original loader target order exactly reproduced the stored U00 seed2041 metric. Therefore this package preserves historical SG selection semantics and provides a separately named official rescoring command instead of silently changing historical validation rankings.

Checkpoint tensor keys are preserved. The public evaluator translates only known legacy factory module names in checkpoint metadata and checks model factory/seed identity. Load only trusted checkpoints: PyTorch2.1 checkpoint loading uses pickle. Outputs must be new directories.

Inference deletion holds weights and anchor generation fixed; it is not an independently retrained missing-field model. HD metrics, where produced by the legacy evaluator, are not MR R@1 and are not the primary reported task.
