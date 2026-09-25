# Model

## Forward path

The model consumes pre-extracted video and query features. Its shared encoder, moment scoring components, and highlight head are trained jointly.

```text
video + query features
  ├─ shared encoder ── shared clip features (384)
  │                      ├─ candidate / boundary heads
  │                      └───────────────────────────────┐
  └─ field encoders ── evidence clip features (64) ───────┤
         ├─ evidence score E                            │
         ├─ support score S                             └─ fusion highlight head
         └─ start/end transition scores T                  └─ one logit per clip

candidate score + E + S + (T_start + T_end) / 2
  └─ ranked temporal windows ── NMS ── moment predictions
```

**E (evidence)** measures query-relevant content. **S (support)** compares the content covered by overlapping temporal ranges, including a centered local residual. **T (transition)** evaluates the start and end of a candidate. The carrier supplies a learned candidate score and boundary predictions. Its scores are centered before the field readout; support anchors are chosen from model predictions without ground-truth input.

The moment score is

```text
score = carrier + E + S + 0.5 * (T_start + T_end)
```

It is a ranking score, not a probability. The highlight logit is a separate output and is not added to this moment score.

## Highlight head

`highlight.py` concatenates shared clip features (384 dimensions) and evidence clip features (64 dimensions):

```text
concat(shared, evidence): 448
  → LayerNorm(448)
  → Linear(448, 64)
  → GELU
  → Linear(64, 1)
```

Both feature paths receive highlight gradients without an extra gradient multiplier. The head is stored under `selector.hd_head` to preserve checkpoint names. Its initialization uses `seed + 9000` in a forked RNG context, leaving backbone initialization and subsequent random draws unchanged.

## Training objectives

All three fields retain their losses. The final objective is

```text
L = 0.5 L_rank
  + 0.1 L_evidence
  + 0.1 L_support
  + 0.1 L_transition
  + 0.1 L_endpoint
  + 0.1 L_highlight
```

| Term | Supervision |
|---|---|
| `L_rank` | Ground-truth temporal quality: candidate-distribution KL, IoU quality calibration, and candidate comparison terms |
| `L_evidence` | GT-overlap mean pooling and reciprocal query/video four-corner hinge (see C02_ARCHITECTURE.md) |
| `L_support` | Coverage-gap hinge for truncated intervals and neutrality for pure outward expansion |
| `L_transition` | Separate start/end temporal-quality supervision |
| `L_endpoint` | Gaussian endpoint targets for the shared boundary head |
| `L_highlight` | Soft VeryGood labels with balanced positives and hard negatives |

The fixed ranking objective uses softened IoU targets in its mixed candidate pool and a training-only scalar calibration intercept. Ranking gradients to the local support gate/readout, support projection, and support edge head are multiplied by 0.1; their auxiliary-loss gradients keep their original scale. The correction has zero forward value and changes only the specified ranking derivatives.

### Highlight targets and hard negatives

For clip `t`, the target is the fraction of its three annotators assigning the highest rating:

```text
y_t = count(rating_t == 4) / 3
```

Targets above zero form the positive group. Targets equal to zero form the negative group, including annotated clips without a VeryGood vote. Padding is excluded using the original annotation duration on the two-second clip grid.

For each query, let `b_t = BCEWithLogits(logit_t, y_t)`:

```text
positive loss = mean(b_t over positive clips)
negative loss = 0.5 * mean(b_t over all negative clips)
              + 0.5 * mean(b_t over the hardest min(16, N_negative) clips)
```

The query loss averages its nonempty positive and negative groups; the batch loss averages valid queries. Empty groups contribute nothing. This fixed configuration does not add an ordinal pairwise highlight loss.

## Implementation map

`joint_model.py` connects moment retrieval and highlight detection. The moment implementation is separated by responsibility:

1. `quality_calibration.py` installs the train-only score intercept and provides the candidate KL helper.
2. `local_support.py` centers carrier inputs, applies the local support residual, and substitutes the softened candidate KL target.
3. `model_factory.py` applies the final coverage objective and support gradient routing.
4. `highlight.py` contains the independent clip head and its loss.

The lower-level field and backbone packages implement feature interaction, span pooling, candidate geometry, and boundary prediction. Internal compatibility names remain where changing them would rename checkpoint tensors. The wrappers preserve parameter construction and floating-point evaluation order; they do not launch experiments or select among alternate recipes.

## Evaluation and checkpoint choice

Training runs for 24 epochs. `best_mr.pt` uses validation Full MR-mAP, then mAP at IoU 0.75, R@1 at IoU 0.7, and the earlier epoch as tie-breakers. `best_hd.pt` uses validation VeryGood HD mAP, then HIT@1, MR-mAP, and the earlier epoch. `last.pt` contains the latest committed epoch, optimizer, scheduler, and RNG states.

The reported joint results use **`best_mr.pt` for both MR and HD**. HD HIT@1 is the official highlight metric and differs from temporal retrieval R@1. Raw saliency logits are evaluated on the real clip grid after padded positions are removed.

The five reported seeds were used for development. Their standard deviation measures observed run variation, not statistical significance or independent confirmation. C02 has already been evaluated on test; its subsequent selection is not a pre-test model lock. Exact component and short-training comparisons check the code cleanup; they do not establish deterministic GPU training.

## Retrieval-only mode

`task="mr"` returns the complete moment model without constructing the highlight module. It has no HD parameters, target construction, loss, prediction scores or checkpoint selector. The E/S/T moment paths remain present. C02 E uses temporal intervals and query/video identities, not saliency ratings. The current loader/time-axis contract remains QV-specific; changing task to mr alone does not implement a Charades or TACoS adapter.
