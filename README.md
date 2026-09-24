# EventFieldNet

Research code for the complete evidence (E), support (S), and transition (T) model. The released recipe is the validation-selected **V00** model: all three fields participate in scoring and have training losses. The backbone and field heads are jointly trained from scratch on fixed, pre-extracted features.

## Installation

Reference environment: Linux, Python 3.10, PyTorch 2.1.0 with CUDA 12.1. Use an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python run.py verify
```

Training on Windows and arbitrary dependency versions is not certified. Source cleanup preserves checkpoint tensor keys; this does not guarantee bitwise deterministic GPU training. Recorded validation scope is in [VALIDATION.md](docs/VALIDATION.md).

## Data

Use SG-DETR's InternVideo2-1B QVHighlights features, not CLIP+SlowFast features. See [DATA.md](DATA.md) for layout and provenance. No dataset, hidden labels, credentials, checkpoint tensors, or raw media are included.

```bash
mkdir -p data
ln -s /absolute/path/to/qvhighlights data/qvhighlights
```

## Train and evaluate

The fixed recipe is `configs/qvhighlights.json`. Seeds 2041-2045 are the five reported runs; each selects its own validation-best checkpoint within 24 epochs. Every command creates a fresh output directory.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke.py --seed 2041 --output runs/smoke2041
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --seed 2041 --output runs/seed2041
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --seed 2041 --checkpoint /path/to/best_val.pt --output runs/val2041
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --seed 2041 --checkpoint /path/to/best_val.pt --split test --output runs/test2041
```

Smoke is four real batch-64 updates plus full validation; never inherit its diagnostic weights into formal training. The training launcher always starts from scratch. Checkpoints are under `runs/seed2041/phase_*/checkpoints/`. Test evaluation must use the checkpoint and configuration already selected on validation; do not use test to select seeds, epochs or recipes.

The training-time SG-derived metric and official Moment-DETR standalone metric are both supported, and must be labelled separately:

```bash
python scripts/score_predictions.py --predictions runs/test2041/full/predictions.json --annotations data/qvhighlights/annotation/highlight_test_with_gt.jsonl --output runs/test2041/moment_detr.json
```

Fixed-checkpoint field deletion (not a retraining ablation):

```bash
CUDA_VISIBLE_DEVICES=0 python run.py delete-fields --seed 2041 --checkpoint /path/to/best_val.pt --output runs/field_deletion2041
```

## Recorded results

| Protocol | Validation Full MR-mAP | Test Full MR-mAP |
|---|---:|---:|
| SG-derived training evaluator | 57.260716 +/- 0.084910 | 55.483140 +/- 0.284554 |
| Official Moment-DETR standalone | not separately reported here | 55.492 +/- 0.280125 |

All values are mean +/- sample SD of five separate validation-selected checkpoints, not an ensemble. They are not last-epoch results. All 55 requested historical checkpoints were tested; the main recipe was selected before test. See [results](results/v00.json) and [EVALUATION.md](docs/EVALUATION.md).

## Code layout

- `src/eventfieldnet/model_factory.py`: EventFieldNet, fixed recipe and rank-gradient routing.
- `src/eventfieldnet/coverage_loss.py`: S truncation hinge and neutral-expansion objective.
- `src/eventfieldnet/model/`: field scoring, counterfactual objectives and probes.
- `src/eventfieldnet/train.py`, `runtime/`: optimizer, schedule and training loop.
- `src/eventfieldnet/precision_evaluation.py`: strict FP32 validation and inference.
- `src/backbone/`, `feature_bridge/`, `field_core/`, `field_primitives/`, `span_fields/`: shared backbone, field encoders and loss foundations.
- `src/training/`: reusable runtime contracts, checkpointing and audits.
- `src/sg_components/`: SG-derived data/metric utilities and inherited EventField backbone modules.
- `external/standalone_eval/`: pinned official Moment-DETR evaluator, with its license.
- `docs/FILE_REVIEW.csv`: file-by-file dependency and cleanup inventory.

Historical deployment launchers and duplicated backbone copies are not shipped. Some internal attribute and checkpoint metadata names remain for compatibility. Lower-level inherited routines needed by initialization, losses or probes are retained rather than replacing the tested method with an unverified rewrite.

Cross-dataset L02 development runs are **not** this V00 model and are not presented as its results. Full retraining ablations and matched cross-dataset evidence remain separate work. Read [LIMITATIONS.md](docs/LIMITATIONS.md) before citing results.
