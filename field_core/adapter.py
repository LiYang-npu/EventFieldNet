"""Adapter that replaces the parent's legacy loss with the direct base.

The reusable parent still owns the encoder, C3 carrier, and candidate grid.
This wrapper owns the only new selector heads and the five-loss contract; it
never calls ``parent_model.compute_loss``.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn

from .losses import compute_five_loss_terms
from .selector import FieldScoreOutput, TriFieldScoreHeads


try:  # The real runner provides these contracts.
    from training.contracts import LossResult, ParameterGroup
except Exception:  # pragma: no cover - local unit tests can run standalone

    @dataclass
    class LossResult:  # type: ignore[no-redef]
        loss: Tensor
        metrics: Mapping[str, Tensor | float]

    @dataclass
    class ParameterGroup:  # type: ignore[no-redef]
        name: str
        params: list[nn.Parameter]
        lr: float
        weight_decay: Optional[float] = None


DEFAULT_PARENT_FACTORY = (
    "backbone.conditioned."
    "trifield_input_conditioner_plugin:build_input_conditioned_trifield_model"
)


def _resolve_target(target: str) -> Any:
    if ":" not in target:
        raise ValueError(f"factory target must be module:attribute, got {target!r}")
    module_name, attribute = target.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def _batch_parts(
    batch: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(batch, Mapping):
        inputs = batch.get("inputs", {})
        targets = batch.get("targets", {})
        metadata = batch.get("metadata", {})
    else:
        inputs = getattr(batch, "inputs", {})
        targets = getattr(batch, "targets", {})
        metadata = getattr(batch, "metadata", {})
    if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("prepared batch inputs and targets must be mappings")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return inputs, targets, metadata


def _masked_softmax(logits: Tensor, valid: Tensor) -> Tensor:
    """Finite masked softmax, including the empty-row case."""

    shape = logits.shape
    flat = logits.float().reshape(logits.shape[0], -1)
    flat_valid = valid.bool().reshape(valid.shape[0], -1)
    safe = flat.masked_fill(~flat_valid, -1.0e4)
    max_value = safe.max(dim=1, keepdim=True).values
    weights = (safe - max_value).exp() * flat_valid.to(safe.dtype)
    normalizer = weights.sum(dim=1, keepdim=True)
    probs = weights / normalizer.clamp_min(1.0e-12)
    return probs.reshape(shape)


def _as_identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    return str(value)


def _metadata_rows(
    metadata: Mapping[str, Any], batch_size: int
) -> list[Mapping[str, Any] | None]:
    """Read the remote loader's ``metadata['metas'][i]`` without assumptions."""

    rows = metadata.get("metas")
    if rows is None:
        rows = metadata.get("records")
    if rows is None:
        # Some test loaders expose the columns directly.
        qids, vids = metadata.get("qid"), metadata.get("vid", metadata.get("video_id"))
        if qids is not None and vids is not None:
            result = []
            for i in range(batch_size):
                result.append(
                    {
                        "qid": qids[i]
                        if isinstance(qids, (list, tuple))
                        else qids[i].item(),
                        "vid": vids[i]
                        if isinstance(vids, (list, tuple))
                        else vids[i].item(),
                    }
                )
            return result
        return [None] * batch_size
    if isinstance(rows, Mapping):
        rows = [rows] * batch_size
    if not isinstance(rows, (list, tuple)):
        return [None] * batch_size
    result = []
    for row in rows[:batch_size]:
        result.append(row if isinstance(row, Mapping) else None)
    result.extend([None] * (batch_size - len(result)))
    return result


def _valid_wrong_query_pairs(
    metadata: Mapping[str, Any], batch_size: int
) -> tuple[Tensor, Tensor]:
    """Choose explicit different query/video rows; never manufacture negatives."""

    rows = _metadata_rows(metadata, batch_size)
    permutation = torch.arange(batch_size, dtype=torch.long)
    pair = torch.zeros(batch_size, dtype=torch.bool)
    for i, row in enumerate(rows):
        if row is None:
            continue
        qi = _as_identifier(row.get("qid", row.get("query_id")))
        vi = _as_identifier(row.get("vid", row.get("video_id")))
        if qi is None or vi is None:
            continue
        # A fixed forward search avoids repeatedly selecting row zero while
        # keeping pair construction deterministic and auditable.
        for j in range(i + 1, len(rows)):
            other = rows[j]
            if other is None:
                continue
            qj = _as_identifier(other.get("qid", other.get("query_id")))
            vj = _as_identifier(other.get("vid", other.get("video_id")))
            if qj is not None and vj is not None and qi != qj and vi != vj:
                permutation[i] = j
                pair[i] = True
                break
    return permutation, pair


class TriFieldBaseModel(nn.Module):
    """C3 carrier plus the direct E/S/T selector and five losses."""

    def __init__(
        self,
        parent_model: nn.Module,
        *,
        selector_hidden_dim: int = 64,
        selector_lr: float = 1.0e-4,
        parent_factory: str = DEFAULT_PARENT_FACTORY,
    ) -> None:
        super().__init__()
        if not isinstance(parent_model, nn.Module):
            raise TypeError("parent factory must return torch.nn.Module")
        self.parent_model = parent_model
        self.selector = TriFieldScoreHeads(
            input_dim=512, hidden_dim=selector_hidden_dim
        )
        self.selector_lr = float(selector_lr)
        if self.selector_lr < 0.0:
            raise ValueError("selector_lr must be non-negative")
        self.parent_factory = str(parent_factory)
        self.loss_weights = {
            "rank": 1.0,
            "evidence": 0.1,
            "support": 0.1,
            "transition": 0.1,
            "endpoint": 0.1,
        }
        self._last_inputs: Any = None
        self._last_field_output: FieldScoreOutput | None = None
        self._last_evidence_pairs = {
            "pair_count": 0,
            "row_count": 0,
            "pair_rate": 0.0,
            "no_pair_rate": 1.0,
        }
        parent_audit = getattr(parent_model, "initialization_audit", {})
        self.initialization_audit = (
            dict(parent_audit) if isinstance(parent_audit, Mapping) else {}
        )
        self.initialization_audit.update(
            {
                "new_base": "field_core",
                "parent_factory": self.parent_factory,
                "checkpoint_inheritance": False,
                "ema": False,
                "projection": False,
                "veto": False,
                "surrogate": False,
                "hardgate": False,
                "frozen_deployment_parameters": False,
                "loss_weights": dict(self.loss_weights),
            }
        )

    def forward(
        self,
        inputs: Any,
        query_features: Optional[Tensor] = None,
        video_padding_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
    ) -> Any:
        # The parent API receives one input mapping.  Keeping this call direct
        # preserves its candidate coordinates and membership exactly.
        output = self.parent_model(
            inputs,
            query_features=query_features,
            video_padding_mask=video_padding_mask,
            query_padding_mask=query_padding_mask,
        )
        extension = getattr(output, "extension_output", None)
        state = getattr(extension, "state", None)
        if state is None:
            raise RuntimeError("reusable parent did not expose InputTriFieldState")
        valid = output.span_valid_mask.bool()
        field = self.selector(
            state,
            valid,
            output.span_logits,
            video_padding_mask=(
                inputs.get("video_padding_mask")
                if isinstance(inputs, Mapping)
                else None
            ),
        )
        # Keep the exact FP32 deployment score used by L_rank; casting back to
        # BF16 here would let loss and ranking see different tie/order values.
        output.span_logits = field.score.masked_fill(~valid, -1.0e4)
        output.span_probs = _masked_softmax(output.span_logits, valid)
        setattr(output, "trifield_output", field)
        self._last_inputs = inputs
        self._last_field_output = field
        return output

    def _wrong_query_evidence(
        self, batch: Any, outputs: Any
    ) -> tuple[Tensor | None, Tensor]:
        """Re-run the actual parent conditioner/backbone for a real wrong query."""

        inputs, _, metadata = _batch_parts(batch)
        source_text = inputs.get("src_txt")
        if not isinstance(source_text, Tensor):
            self._last_evidence_pairs = {
                "pair_count": 0,
                "row_count": int(outputs.span_logits.shape[0]),
                "pair_rate": 0.0,
                "no_pair_rate": 1.0,
            }
            return None, torch.zeros(
                outputs.span_logits.shape[0],
                dtype=torch.bool,
                device=outputs.span_logits.device,
            )
        permutation_cpu, pair_cpu = _valid_wrong_query_pairs(
            metadata, source_text.shape[0]
        )
        permutation, pair = (
            permutation_cpu.to(source_text.device),
            pair_cpu.to(source_text.device),
        )
        pair_count = int(pair.sum().item())
        self._last_evidence_pairs = {
            "pair_count": pair_count,
            "row_count": int(pair.numel()),
            "pair_rate": float(pair.float().mean().item()) if pair.numel() else 0.0,
            "no_pair_rate": float((~pair).float().mean().item())
            if pair.numel()
            else 1.0,
        }
        if pair_count == 0:
            return None, pair
        wrong_inputs = dict(inputs)
        wrong_inputs["src_txt"] = source_text.index_select(0, permutation)
        query_pad = inputs.get("query_padding_mask")
        if isinstance(query_pad, Tensor):
            wrong_inputs["query_padding_mask"] = query_pad.index_select(0, permutation)
        wrong_output = self.parent_model(wrong_inputs)
        wrong_extension = getattr(wrong_output, "extension_output", None)
        wrong_state = getattr(wrong_extension, "state", None)
        if wrong_state is None:
            raise RuntimeError(
                "wrong-query parent forward did not expose InputTriFieldState"
            )
        wrong_field = self.selector.raw_from_state(
            wrong_state,
            outputs.span_valid_mask.bool(),
            wrong_output.span_logits,
            video_padding_mask=inputs.get("video_padding_mask"),
        )
        return wrong_field.evidence, pair

    def compute_loss(
        self, outputs: Any, batch: Any, teacher_outputs: Any, epoch: int
    ) -> LossResult:
        if teacher_outputs is not None:
            raise AssertionError(
                "field_core is scratch-only and forbids teacher/KD fusion"
            )
        wrong_evidence, pair = self._wrong_query_evidence(batch, outputs)
        terms = compute_five_loss_terms(
            outputs,
            batch,
            wrong_evidence=wrong_evidence,
            evidence_pair_mask=pair,
        )
        metrics = dict(terms.metrics)
        metrics.update(
            {
                "epoch": float(epoch),
                "evidence/pair_rate": self._last_evidence_pairs["pair_rate"],
                "evidence/no_pair_rate": self._last_evidence_pairs["no_pair_rate"],
            }
        )
        return LossResult(terms.total, metrics)

    def diagnostics(self, outputs: Any, batch: Any) -> Mapping[str, Tensor | float]:
        result: dict[str, Tensor | float] = {}
        parent_diagnostics = getattr(self.parent_model, "diagnostics", None)
        if callable(parent_diagnostics):
            result.update(parent_diagnostics(outputs, batch))
        field = getattr(outputs, "trifield_output", None)
        if field is None:
            return result
        valid = field.valid
        selected = lambda value: (
            value[valid] if bool(valid.any()) else value.reshape(-1)[:0]
        )
        for name, value in (
            ("carrier_raw", field.carrier_raw),
            ("carrier", field.carrier),
            ("evidence", field.evidence),
            ("support", field.support),
            ("transition_start", field.transition_start),
            ("transition_end", field.transition_end),
            ("score", field.score),
        ):
            values = selected(value).float()
            if values.numel():
                result[f"trifield/{name}_mean"] = values.mean().detach()
                result[f"trifield/{name}_p01"] = torch.quantile(values, 0.01).detach()
                result[f"trifield/{name}_p99"] = torch.quantile(values, 0.99).detach()
                result[f"trifield/{name}_finite"] = (
                    torch.isfinite(values).all().detach()
                )
        state = getattr(getattr(outputs, "extension_output", None), "state", None)
        if state is not None:
            for name in (
                "evidence_score",
                "support_score",
                "enter_score",
                "leave_score",
            ):
                value = getattr(state, name, None)
                if isinstance(value, Tensor):
                    result[f"trifield/state_{name}_mean"] = (
                        value.float().mean().detach()
                    )
            result["trifield/role_updates_norm"] = (
                state.role_updates.float().norm(dim=-1).mean().detach()
            )
        result["trifield/evidence_pair_rate"] = float(
            self._last_evidence_pairs["pair_rate"]
        )
        result["trifield/evidence_no_pair_rate"] = float(
            self._last_evidence_pairs["no_pair_rate"]
        )
        return result

    def set_epoch(self, epoch: int, training: bool) -> Mapping[str, Any]:
        setter = getattr(self.parent_model, "set_epoch", None)
        result = dict(setter(epoch, training)) if callable(setter) else {}
        result.update(
            {"base": "field_core", "epoch": int(epoch), "training": bool(training)}
        )
        return result

    @staticmethod
    def _iou(outputs: Any, batch: Any) -> Tensor:
        # Compatibility helper; the five losses use the richer geometry object.
        from .losses import candidate_geometry

        return candidate_geometry(outputs, batch).max_iou

    def decode(self, outputs: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        decoder = getattr(self.parent_model, "decode", None)
        if callable(decoder):
            return decoder(outputs, inputs, *args, **kwargs)
        # The verified parent is a score shell and exposes no decoder.  This
        # finite argmax fallback follows the Stage47 contract; official eval
        # converts the selected grid index with the unchanged coordinates.
        valid = outputs.span_valid_mask.bool()
        return (
            outputs.span_logits.float().masked_fill(~valid, -1.0e4).flatten(1).argmax(1)
        )

    def parameter_groups(self) -> list[ParameterGroup]:
        parent_groups_fn = getattr(self.parent_model, "parameter_groups", None)
        groups = list(parent_groups_fn()) if callable(parent_groups_fn) else []
        used = {id(parameter) for group in groups for parameter in group.params}
        selector_params = [
            p
            for p in self.selector.parameters()
            if p.requires_grad and id(p) not in used
        ]
        if selector_params:
            groups.append(
                ParameterGroup("trifield_heads", selector_params, self.selector_lr)
            )
            used.update(id(parameter) for parameter in selector_params)
        missing = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and id(parameter) not in used
        ]
        if missing:
            raise RuntimeError(f"optimizer parameter group omission: {missing}")
        return groups

    def trainability_audit(self) -> Mapping[str, Any]:
        parent_audit_fn = getattr(self.parent_model, "trainability_audit", None)
        parent_audit = dict(parent_audit_fn()) if callable(parent_audit_fn) else {}
        frozen_deployment: list[str] = []
        inactive: list[str] = []
        inactive_test = getattr(
            self.parent_model, "is_full_path_inactive_parameter", None
        )
        inactive_prefixes = (
            "stem.evidence_readout.",
            "stem.support_readout.",
            "stem.start_transition_readout.",
            "stem.end_transition_readout.",
            "stem.transition_readout.",
            "evidence_modulation.",
            "support_role.",
            "support_pyramid.",
            "transition_adapter.",
            "neural_quality_field.",
            "context_contrast.",
            "matched_head.",
        )
        for name, parameter in self.parent_model.named_parameters():
            if parameter.requires_grad:
                continue
            if callable(inactive_test) and inactive_test(name):
                inactive.append(name)
            elif any(
                name.startswith(prefix) or prefix in name
                for prefix in inactive_prefixes
            ):
                inactive.append(name)
            elif "support_residual_logit" in name:
                inactive.append(name)
            else:
                frozen_deployment.append(name)
        parent_audit.update(
            {
                "new_base_selector_trainable": all(
                    p.requires_grad for p in self.selector.parameters()
                ),
                "inactive_compatibility_parameters": inactive,
                "frozen_deployment_parameters": frozen_deployment,
                "optimizer_group_coverage": True,
            }
        )
        return parent_audit

    def experiment_contract(self) -> Mapping[str, Any]:
        contract_fn = getattr(self.parent_model, "experiment_contract", None)
        result = dict(contract_fn()) if callable(contract_fn) else {}
        result.update(
            {
                "base": "field_core",
                "initialization": "scratch_random_initialization",
                "checkpoint_inheritance": False,
                "original_coordinates": True,
                "candidate_membership_unchanged": True,
                "final_score": "tanh(carrier_raw)+E+S+0.5*(Ts+Te)",
                "carrier_bound": "tanh(carrier_raw), range[-1,1]",
                "evidence": "tanh(raw_evidence-mean_valid_candidate(raw_evidence))",
                "support": "tanh(raw_support)",
                "transition": "tanh(raw_transition_start), tanh(raw_transition_end)",
                "loss_terms": {
                    "rank": "official all-candidate ordinal zero-margin",
                    "evidence": "positive IoU>=0.70 true-vs-explicit-wrong query hinge(0.20)",
                    "support": "balanced coverage tolerant square, target min(recall,precision)",
                    "transition": "same-GT endpoint quality tolerant square",
                    "endpoint": "original Gaussian start/end BCE",
                },
                "loss_weights": dict(self.loss_weights),
                "ordinal_ties": "zero scalar at tie with separating subgradient convention",
                "multi_gt_assignment": "max IoU, stable first-index tie; duplicate GTs are not unique-matched",
                "wrong_query_policy": "requires metadata qid and vid both different; no synthetic roll negatives",
                "old_losses_removed": [
                    "reader",
                    "rank2",
                    "top1_router",
                    "anchor",
                    "semantic_preserve",
                    "saliency",
                    "hardrank",
                    "spanset",
                    "surrogate",
                    "veto",
                    "projection",
                ],
            }
        )
        return result


def build_trifield_base_model(
    config: Any = None,
    *,
    parent_factory: str = DEFAULT_PARENT_FACTORY,
    selector_hidden_dim: int = 64,
    selector_lr: float = 1.0e-4,
    **kwargs: Any,
) -> TriFieldBaseModel:
    """Build the new base around the verified reusable parent factory."""

    factory = (
        _resolve_target(parent_factory)
        if isinstance(parent_factory, str)
        else parent_factory
    )
    inspect.signature(factory)
    # New-base knobs are consumed here.  Remaining kwargs are deliberately
    # passed to the parent because its existing builder owns the Stage32/C3
    # configuration contract.
    parent_model = factory(config, **kwargs)
    return TriFieldBaseModel(
        parent_model,
        selector_hidden_dim=selector_hidden_dim,
        selector_lr=selector_lr,
        parent_factory=parent_factory,
    )


__all__ = ["DEFAULT_PARENT_FACTORY", "TriFieldBaseModel", "build_trifield_base_model"]
