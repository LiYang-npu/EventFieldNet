# EventFieldNet — C02

中文入口：[论文阅读指南](docs/论文阅读指南.md)。

Start with [the architecture and equations](docs/C02_ARCHITECTURE.md) and [the code map](docs/C02_CODE_MAP.md). This directory is independent of the preserved F05 release. See [verification scope](docs/C02_VERIFICATION.md).



EventFieldNet jointly learns temporal moment retrieval (MR) and query-based highlight detection (HD). Evidence, support, and transition fields contribute to moment scores and retain their corresponding training losses. A separate highlight head fuses shared video features with evidence features.

This release contains the fixed **C02** configuration (four-corner counterfactual evidence supervision). It uses SG-DETR's released InternVideo2-1B features and trains the EventFieldNet model from scratch.

## Setup

Reference environment: Python 3.10, PyTorch 2.1.0, CUDA 12.1, Linux.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the QVHighlights annotations and the [SG-DETR InternVideo2-1B features](https://github.com/ai-forever/sg-detr), then arrange them as follows:

```text
data/qvhighlights/
  annotation/
    highlight_train_release.jsonl
    highlight_val_release.jsonl
  custom_features/
    video/<video_id>.pt
    custom_text/<query_id>.npz
```

The video and text feature dimensions are 512; temporal endpoint features add two video channels. Use the original two-second clip grid and annotation durations. The data loader handles padding internally.

## Task configuration

The default `configs/qvhighlights.json` uses `"task": "mr_hd"` and reproduces the fixed F05 model.

For retrieval without a highlight task, use `configs/qvhighlights_mr.json` (`"task": "mr"`). This creates no highlight head, evaluates only MR and writes no `best_hd.pt`. It retains all three moment fields and their losses. Do not use a zero HD-loss weight as a substitute for disabling the branch.

The task switch is separate from dataset adaptation: F05's E objective uses genuine QV ordinal ratings even when HD is disabled. Datasets without these ratings need their own validated E target adapter; missing ratings raise an error rather than silently disabling E. Charades/TACoS adapters are not part of this fixed QV release. See [task modes](docs/TASKS.md).

## Train

Run commands from the repository root. Each fresh run requires a new output directory.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/qvhighlights.json \
  --seed 2041 --output runs/seed2041
```

The fixed budget is 24 epochs, batch size 64, with three warmup epochs and learning-rate milestones at 10, 20, and 40. Training and evaluation use FP32 with TF32 disabled. The five reported seeds are **2041, 2042, 2043, 2044, 2045**.

To resume a committed epoch from the same run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/qvhighlights.json \
  --seed 2041 --output runs/seed2041 \
  --resume runs/seed2041/last.pt
```

The output directory contains `history.json`, `best.json`, `best_mr.pt`, `best_hd.pt`, and `last.pt`. The MR and HD selectors are separate; use `best_mr.pt` to reproduce the joint results below.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config configs/qvhighlights.json \
  --checkpoint runs/seed2041/best_mr.pt \
  --output runs/val2041
```

Inference reports MR and HD from the same checkpoint. MR uses the preserved SG-derived retrieval protocol; HD uses the pinned official QVHighlights evaluator, including removal of padded clips. `inference.py` currently evaluates the validation split.

## Memory efficiency

Long-video training automatically recomputes large candidate activations and releases completed batch tensors. Historical F05 measurements are retained for context; those savings have not been remeasured for the extra C02 E counterfactual branches. See [measurement conditions and checks](docs/EFFICIENCY.md).

## Recorded C02 results

Original sealed C02 experiments: validation MR-best mean Full mAP 57.463646 (sample SD 0.198993), same-checkpoint HIT@1 73.304 and HD mAP 43.840, across seeds 2041-2045. Test official all-GT Full mAP is 55.844 (sample SD 0.494), HIT@1 71.038 and HD mAP 43.408. The SG-derived test protocol instead gives Full mAP 55.833; do not mix evaluator protocols.

These are original experiment results, not a new training run of this reorganized package. All metrics use the per-seed validation-MR-best checkpoint. The seeds are observed development seeds. C02 selection occurred after test results were available.

## Code layout

| Path | Purpose |
|---|---|
| `train.py`, `inference.py` | Training and validation entry points |
| `configs/qvhighlights.json` | Task selection and thirteen training options |
| `eventfieldnet/recipe.json` | Fixed model, data and loss construction settings |
| `eventfieldnet/joint_model.py` | Complete moment model plus highlight branch |
| `eventfieldnet/highlight.py` | Fusion head and balanced hard-negative BCE |
| `eventfieldnet/model_factory.py` | Three-field objective and ranking-gradient routing |
| `eventfieldnet/local_support.py`, `coverage_loss.py` | Local support readout and coverage supervision |
| `eventfieldnet/engine.py` | Training loop, checkpointing, and selection |
| `eventfieldnet/evaluation.py`, `hd_evaluator.py` | Checkpoint reconstruction and MR/HD metrics |
| `backbone/`, `field_core/`, `field_primitives/`, `span_fields/` | Shared encoder and field components |
| `external/standalone_eval/` | Pinned official QVHighlights evaluator |

The model defines explicit AdamW parameter-group learning rates; `learning_rate` is their fallback, not a multiplier for every group. Each run writes its actual groups to `optimizer.json`.

Run the C02 objective regression suite with `python -m unittest discover -s tests -v`. See [verification details](docs/VERIFICATION.md) and the [file guide](docs/FILES.md).

See [the model description](docs/MODEL.md) for the scoring path and losses. Some internal tensor names are retained to load existing checkpoints exactly. Third-party provenance and licenses are listed in `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSE.txt`.
