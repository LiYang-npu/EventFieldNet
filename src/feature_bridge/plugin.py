"""Repository factories for Stage50."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Optional
import torch
from torch import nn
from training.config import RunnerConfig
from training.contracts import DataBundle, PreparedBatch
from scripts import event_field_f0_train as f0
from sg_components.dataset.collate import move_inputs_to_device
from .evaluator import SegmentOfficialEvalAdapter
from .model import SemanticAnchoredAtomGraph

STAGE32_KWARGS = dict(
    video_dim=514,
    query_dim=512,
    hidden_dim=384,
    num_heads=8,
    interaction_layers=3,
    feedforward_dim=1536,
    pair_dim=96,
    dropout=0.2,
    min_span_clips=2,
    context_rank=96,
    context_scales=(1,),
    context_residual_bound=1.0,
    context_boundary_radius=4,
    use_context_quality=True,
    boundary_rank=96,
    boundary_scales=(1, 3, 7),
    boundary_residual_bound=1.0,
    use_boundary_gate=True,
    field_rank=64,
    use_evidence_modulation=True,
    use_support_pyramid=True,
    use_derived_transition=True,
    use_quality_energy=True,
    bounded_field_energy=True,
    detach_quality_features=True,
    anchor_preserving_span=True,
    structured_energy_bound=0.5,
    quality_energy_bound=2.0,
)


def checkpoint_state(path):
    payload = torch.load(Path(path), map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint must contain model state")
    return state


def initialize_from_stage32(model: nn.Module, path):
    state = checkpoint_state(path)
    expected = set(model.state_dict()) - set(state)
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(f"Stage32 additive init mismatch: {incompatible}")
    return {
        "checkpoint": str(path),
        "missing_keys": sorted(expected),
        "unexpected_keys": [],
        "strict_additive": True,
    }


def build_atom_model(
    config: RunnerConfig,
    mode: str,
    init_checkpoint: str,
    atom_lr=1e-4,
    atom_rank=48,
    atom_hidden=96,
    num_atoms=8,
):
    del config
    model = SemanticAnchoredAtomGraph(
        **STAGE32_KWARGS,
        mode=mode,
        atom_lr=atom_lr,
        atom_rank=atom_rank,
        atom_hidden=atom_hidden,
        num_atoms=num_atoms,
    )
    model.initialization_audit = initialize_from_stage32(model, init_checkpoint)
    return model


def _prepare(raw: Any, device: torch.device):
    metas, raw_batch = raw
    inputs, labels = move_inputs_to_device(raw_batch, device)
    if labels is None:
        raise RuntimeError("labels missing")
    spans, mask = f0.pad_spans(labels["span_labels"], device)
    return PreparedBatch(
        inputs={
            "src_vid": inputs["src_vid"],
            "src_txt": inputs["src_txt"],
            "video_padding_mask": ~inputs["src_vid_mask"].bool(),
            "query_padding_mask": ~inputs["src_txt_mask"].bool(),
        },
        targets={"gt_spans": spans, "gt_span_mask": mask},
        metadata={"metas": metas},
        batch_size=int(inputs["src_vid"].shape[0]),
    )


def build_repository_data(config: RunnerConfig, root: str):
    root = Path(root)
    return DataBundle(
        train_loader=f0.make_loader(
            f0.make_dataset(root, "train"), config.batch_size, config.num_workers, True
        ),
        val_loader=f0.make_loader(
            f0.make_dataset(root, "val"), config.batch_size, config.num_workers, False
        ),
        prepare_batch=_prepare,
        metadata={"root": str(root), "protocol": "qvhighlights-fixed-final"},
    )


def build_official_evaluator(
    config: RunnerConfig, root: str, project_root: str, wrapper: Optional[str] = None
):
    del config
    path = (
        Path(wrapper)
        if wrapper
        else Path(root) / "stage50" / "feature_bridge" / "official_eval.py"
    )
    return SegmentOfficialEvalAdapter(
        wrapper=str(path), project_root=str(project_root), root=str(root)
    )
