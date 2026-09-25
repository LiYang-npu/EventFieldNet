"""EventFieldNet with complete E/S/T scoring and a fusion highlight head."""

import json
from pathlib import Path

import torch

from field_core.adapter import _batch_parts, _metadata_rows
from training.contracts import LossResult

from .highlight import HighlightHead, highlight_loss
from .model_factory import EventFieldNet as MomentModel
from .model_factory import build_model as build_moment_model


def _require_evidence_annotations(output, batch):
    """Reject missing ordinal E supervision without requiring active pairs."""
    message = (
        "MR-only disables HD; the fixed E objective requires ordinal annotations. "
        "Use a dataset-specific E target adapter for unrated data."
    )
    _, targets, metadata = _batch_parts(batch)
    token = output.trifield_output.round31_e_token
    ratings = targets.get("saliency_all_labels")
    if not isinstance(ratings, torch.Tensor) or ratings.shape != token.shape:
        raise ValueError(message + " Expected saliency_all_labels with shape [B, L].")
    if not torch.isfinite(ratings).all():
        raise ValueError(message + " Ordinal ratings must be finite.")
    for row in _metadata_rows(metadata, token.shape[0]):
        if row is None or "relevant_clip_ids" not in row or row["relevant_clip_ids"] is None:
            raise ValueError(message + " Missing explicit relevant_clip_ids metadata.")
        try:
            ids = torch.as_tensor(row["relevant_clip_ids"])
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(message + " Invalid relevant_clip_ids metadata.") from error
        if (
            ids.ndim != 1 or not torch.isfinite(ids).all()
            or not (ids == ids.long()).all()
            or not ((ids >= 0) & (ids < token.shape[1])).all()
        ):
            raise ValueError(message + " Rated clip IDs must be in-range integers.")


class RetrievalModel(MomentModel):
    """Complete three-field MR without an HD head or highlight objective."""

    def compute_loss(self, output, batch, teacher_outputs, epoch):
        _require_evidence_annotations(output, batch)
        return super().compute_loss(output, batch, teacher_outputs, epoch)


class EventFieldNet(MomentModel):
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output.hd_source = torch.cat(
            (
                output.token_features,
                output.trifield_output.round31_e_readout_input_prime,
            ),
            dim=-1,
        )
        output.hd_logits = self.selector.hd_head(output.hd_source)
        return output

    def highlight_targets(self, output, batch):
        """Build soft VeryGood targets on the original two-second clip grid."""
        _, _, metadata = _batch_parts(batch)
        rows = _metadata_rows(metadata, output.hd_logits.shape[0])
        target = torch.zeros_like(output.hd_logits)
        mask = torch.zeros_like(target, dtype=torch.bool)
        for index, row in enumerate(rows):
            if row is None:
                raise ValueError("Highlight training requires query metadata")
            qid = str(row["qid"])
            clip_count = self.highlight_lengths[qid]
            if not 0 < clip_count <= target.shape[1]:
                raise ValueError(f"Invalid highlight clip count for query {qid}")
            mask[index, :clip_count] = True
            ids = torch.as_tensor(
                row["relevant_clip_ids"], device=target.device, dtype=torch.long
            )
            ratings = torch.as_tensor(
                row["saliency_scores"], device=target.device, dtype=torch.float32
            )
            if ratings.shape != (len(ids), 3) or not (
                (ratings >= 0) & (ratings <= 4)
            ).all():
                raise ValueError(f"Invalid three-annotator ratings for query {qid}")
            if not ((ids >= 0) & (ids < clip_count)).all() or (
                ids.unique().numel() != ids.numel()
            ):
                raise ValueError(f"Invalid or duplicate highlight clips for query {qid}")
            target[index, ids] = (ratings >= 4).float().mean(1)
        return target, mask

    def compute_loss(self, output, batch, teacher_outputs, epoch):
        moment = super().compute_loss(output, batch, teacher_outputs, epoch)
        target, mask = self.highlight_targets(output, batch)
        highlight, parts = highlight_loss(output.hd_logits, target, mask)
        total = moment.loss + 0.1 * highlight
        metrics = dict(moment.metrics)
        metrics.update({"hd/" + name: value.detach() for name, value in parts.items()})
        metrics.update(
            {
                "hd/loss": highlight.detach(),
                "hd/weighted_loss": 0.1 * highlight.detach(),
                "hd/logit_std": output.hd_logits[mask].std(unbiased=False).detach(),
                "total": total.detach(),
            }
        )
        return LossResult(total, metrics)

    def experiment_contract(self):
        contract = dict(super().experiment_contract())
        contract["highlight"] = {
            "input": "shared_384_plus_evidence_64",
            "head": "LayerNorm448_Linear64_GELU_Linear1",
            "target": "fraction_of_three_VeryGood_ratings",
            "loss": "balanced_soft_BCE_all_and_top16_negatives",
            "loss_weight": 0.1,
            "backbone_gradient_scale": 1.0,
            "initialization_seed_offset": 9000,
        }
        return contract


def build_model(config, *, annotation_files=None, enable_highlight=True, **kwargs):
    """Build complete three-field MR, with the optional fusion highlight head.

    With highlights disabled, no HD parameters, labels, loss, or metadata are
    constructed. The fixed E training objective still requires ordinal ratings. With highlights enabled, annotation lengths mask only the
    training loss; forward inference does not consult labels or these lengths.
    """
    if type(enable_highlight) is not bool:
        raise TypeError("enable_highlight must be a boolean")
    if enable_highlight and not annotation_files:
        raise ValueError("Highlight training requires annotation_files")
    model = build_moment_model(config, **kwargs)
    if not enable_highlight:
        model.__class__ = RetrievalModel
        return model
    model.__class__ = EventFieldNet
    model.highlight_lengths = {}
    for filename in annotation_files:
        with Path(filename).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["qid"])
                if qid in model.highlight_lengths:
                    raise ValueError(f"Duplicate annotation query {qid}")
                model.highlight_lengths[qid] = int(row["duration"] / 2)

    # The head neither consumes nor changes the backbone initialization RNG.
    seed = config["seed"] if isinstance(config, dict) else config.seed
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 9000)
        model.selector.hd_head = HighlightHead()
    return model
