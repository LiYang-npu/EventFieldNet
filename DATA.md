# Data and features

Use the annotations and **InternVideo2-1B features released by SG-DETR**:

- [SG-DETR feature instructions](https://github.com/ai-forever/sg-detr)
- [QVHighlights annotations and evaluation](https://github.com/jayleicn/moment_detr)

See the directory layout in README.md. Video features are `<vid>.pt`; text features are `<qid>.npz` with a `features` array. Both have 512 channels. Video temporal endpoint features add two channels. The loader preserves normalization, maximum query length 40, maximum video length 75 and the two-second clip grid.

The validation split contains 1550 queries. The included annotation byte hashes are in `docs/annotation_sha256.json`; feature files are external and are not authenticated by annotation hashes. Features and model checkpoints are not bundled.

The task model is initialized from scratch. Using pretrained InternVideo2 features is separate from additional task-model pretraining; this fixed configuration loads no task checkpoint at startup.

The shipped CLI evaluates validation. No F05 test result is claimed, and test labels must not replace validation annotations. Third-party licenses and dataset terms apply.
