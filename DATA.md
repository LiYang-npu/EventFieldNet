# Data and features

```
data/qvhighlights/
  annotation/highlight_train_release.jsonl
  annotation/highlight_val_release.jsonl
  annotation/highlight_test_with_gt.jsonl  # optional, evaluation only
  custom_features/video/<vid>.pt
  custom_features/custom_text/<qid>.npz
```

Video and text: SG-DETR released InternVideo2-1B features, 512 channels each; video TEF adds two channels. Preserve max query length40, max video length75, clip length2 seconds, normalization and the reference coordinate convention. The same feature family is used for train, val and test. Feature extraction with a pretrained backbone is distinct from additional InterVid-MR task-model pretraining; this recipe does not load a pretrained EventFieldNet checkpoint.

Official sources: https://github.com/ai-forever/sg-detr (features), https://github.com/jayleicn/moment_detr (QVHighlights annotations). Respect their licenses and dataset terms. The package does not download third-party data automatically. Train/val byte hashes are checked using docs/annotation_sha256.json. Feature contents are external and are not authenticated by annotation hashes.

Val has1550 queries; heldout test has1542. The test GT release used for recorded results is pinned to Moment-DETR commit b7e553ac3b0c898ee6b85e03ee507c064eab89ca; file SHA256 bd50ec6bb5dd3f72571126ba5fdc7418efdd9219a9487984e483a02e2ce5d493. No test label is passed as a model input. Do not substitute test data into validation filenames.

configs/probe_panel.json contains public training query IDs for diagnostics. Training/validation probe observations are not additional optimization targets.
