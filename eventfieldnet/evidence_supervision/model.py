"""F05 with a separately configured, rating-free local E objective."""
import dataclasses
import json
import math
from pathlib import Path
import torch
from field_core.adapter import _batch_parts, _metadata_rows
from eventfieldnet import joint_model
from eventfieldnet.model_factory import EventFieldNet as MomentModel
from .pairing import reciprocal_pairs
from .e_only import evidence_tokens
from .objectives import objective


def annotation_clip_counts(annotation_files):
    """Original QV seconds, before the loader substitutes FPN-padded duration."""
    counts = {}
    for filename in annotation_files:
        for line in Path(filename).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["qid"])
            duration = float(row["duration"])
            if key in counts:
                raise ValueError("Duplicate QID")
            if not math.isfinite(duration) or not 2 <= duration <= 150:
                raise ValueError(f"QV duration outside the audited 75-clip axis: {key}")
            counts[key] = int(duration / 2)
    return counts


def audit_annotation_features(annotation_files):
    """Audit distinct annotation and feature axes; never edit either one.

    SG's supplied QV feature sequence contains annotation_clips or one extra
    nonzero terminal feature. The loader caps features at 75 and then pads to
    FPN4. Only annotation-covered clips participate in the new E supervision.
    """
    counts = annotation_clip_counts(annotation_files)
    checked = {}
    annotation_counts_by_feature = {}
    mismatch_queries = []
    raw_offsets = {}
    loader_offsets = {}
    fpn_offsets = {}
    for filename in annotation_files:
        feature_root = Path(filename).parent.parent / "custom_features" / "video"
        for line in Path(filename).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            real = counts[qid]
            path = feature_root / (str(row["vid"]) + ".pt")
            key = str(path.resolve())
            if key not in checked:
                feature = torch.load(path, map_location="cpu")
                if not isinstance(feature, torch.Tensor) or feature.ndim != 2 or feature.shape[1] != 512:
                    raise ValueError(f"Expected QV [feature_clips,512] tensor: {path}")
                checked[key] = int(feature.shape[0])
            raw = checked[key]
            used = min(raw, 75)
            padded = ((used + 3) // 4) * 4
            if raw - real not in (0, 1):
                raise ValueError(f"Feature count outside audited annotation or annotation+1 contract: {qid}")
            if key in annotation_counts_by_feature:
                if annotation_counts_by_feature[key] != real:
                    raise ValueError(f"Conflicting annotation durations for one video: {qid}")
            else:
                annotation_counts_by_feature[key] = real
                for histogram, offset in ((raw_offsets, raw-real),
                                          (loader_offsets, used-real),
                                          (fpn_offsets, padded-real)):
                    histogram[str(offset)] = histogram.get(str(offset), 0) + 1
            if used != real:
                mismatch_queries.append({"qid": qid, "vid": str(row["vid"]),
                    "annotation_clips": real, "raw_feature_clips": raw,
                    "loader_feature_clips": used, "fpn_clips": padded})
    return {"passed": True, "queries": len(counts), "unique_features": len(checked),
            "clip_seconds": 2, "max_annotation_clips": 75, "fpn_multiple": 4,
            "feature_length_cap": 75, "annotation_duration_clipped": False,
            "supervision_axis": "original_annotation_duration_floor_divide_2",
            "normalization_axis": "actual_feature_cap75_then_FPN4",
            "offset_histogram_unit": "unique_feature_file",
            "raw_minus_annotation_histogram": raw_offsets,
            "loader_minus_annotation_histogram": loader_offsets,
            "fpn_minus_annotation_histogram": fpn_offsets,
            "extra_feature_is_background": False,
            "loader_annotation_mismatch_queries": mismatch_queries}


def gt_weights(batch, clip_counts, length):
    inputs, targets, metadata = _batch_parts(batch)
    rows = _metadata_rows(metadata, inputs["src_vid"].shape[0])
    spans=targets["gt_spans"].float(); gm=targets["gt_span_mask"].bool()
    if spans.shape[:2] != gm.shape or spans.shape[-1] != 2:
        raise ValueError("Bad GT shape")
    if not torch.isfinite(spans[gm]).all(): raise ValueError("Nonfinite GT")
    real=torch.tensor([clip_counts[str(r["qid"])] for r in rows],device=spans.device)
    # QV metadata duration is the FPN-padded seconds, as used by gt_spans.
    padded_seconds=torch.tensor([float(r["duration"]) for r in rows],device=spans.device)
    # The annotation axis and feature/FPN axis differ for 71 supplied queries.
    # Full feature audit validates raw_N in {annotation_N, annotation_N+1};
    # at runtime the actual loader metadata and prefix mask identify the FPN axis.
    actual_padded = padded_seconds / 2
    if (not torch.isfinite(actual_padded).all()
            or not (actual_padded == actual_padded.round()).all()
            or not (actual_padded.remainder(4) == 0).all()
            or not ((actual_padded >= real) & (actual_padded <= real + 4)
                    & (actual_padded <= length)).all()):
        raise ValueError("GT normalization duration outside audited feature/FPN axis")
    model_valid = ~inputs["video_padding_mask"].bool()
    expected_valid = torch.arange(length, device=spans.device)[None, :] < actual_padded[:, None]
    if model_valid.shape != expected_valid.shape or not torch.equal(model_valid, expected_valid):
        raise ValueError("Model prefix padding mask differs from actual sample FPN length")
    clip_spans=spans*padded_seconds[:,None,None]/2.0
    if not ((clip_spans[...,0][gm]>=0)&(clip_spans[...,1][gm]>clip_spans[...,0][gm])).all():
        raise ValueError("Invalid GT interval")
    if not ((real>0)&(real<=length)).all(): raise ValueError("Invalid real clip count")
    if (clip_spans[...,1][gm] > real[:,None].expand_as(gm)[gm]+1e-4).any():
        raise ValueError("GT exceeds real feature time axis")
    pos=torch.arange(length,device=spans.device,dtype=spans.dtype)
    overlap=(torch.minimum(clip_spans[...,1,None],pos[None,None,:]+1)-
             torch.maximum(clip_spans[...,0,None],pos[None,None,:])).clamp_min(0)
    real_mask=pos[None,:]<real[:,None]
    overlap=overlap*gm[...,None]*real_mask[:,None,:]
    weights=overlap/overlap.sum(-1,keepdim=True).clamp_min(1e-12)
    return weights,real_mask,rows


class CounterfactualMixin:
    def compute_loss(self, output, batch, teacher_outputs, epoch):
        inputs,_,_= _batch_parts(batch)
        weights, valid_tokens,rows=gt_weights(batch,self.cf_clip_counts,inputs["src_vid"].shape[1])
        pairs,pair_valid,excluded=reciprocal_pairs(rows,inputs["src_vid"].device)
        # Every arm uses the same deterministic E function for both conditions.
        # No raw/encoded video intervention or full parent rerun is needed here.
        v=inputs["src_vid"][...,:512];t=inputs["src_txt"]
        vp=inputs["video_padding_mask"];tp=inputs["query_padding_mask"]
        clean=evidence_tokens(self.selector,v,t,vp,tp)
        wrong=evidence_tokens(self.selector,v,t[pairs],vp,tp[pairs])
        evidence,probe=objective(clean,wrong,weights,pairs,pair_valid,arm=self.cf_arm,margin=.2,temperature=.2)
        probe.update({"pair_rejected/"+k:clean.new_tensor(float(v)) for k,v in excluded.items()})
        probe.update(aux_uses_ordinal_ratings=clean.new_zeros(()),
                     aux_dropout_disabled=clean.new_ones(()),
                     extra_parent_forwards=clean.new_zeros(()),
                     real_clip_count=valid_tokens.sum().float(),
                     fpn_excluded_clip_count=((~vp.bool())&(~valid_tokens)).sum().float())
        output.trifield_output.cf_e_objective=(evidence,probe)
        self.cf_last_evidence=evidence
        self.cf_last_probe={k:v.detach() for k,v in probe.items()}
        result=super().compute_loss(output,batch,teacher_outputs,epoch)
        # The normal gradient paths are unchanged. Probe each real epoch's first
        # batch before backward, including readout vs encoder and S/T separately.
        if self.training and torch.is_grad_enabled() and self.cf_probe_epoch != epoch:
            self.cf_probe_epoch=epoch
            terms=self.r50_last_terms
            groups={"E":(evidence,list(self.selector.e_interaction.parameters())),
                    "S":(terms.support,list(self.selector.s_projection.parameters())+list(self.selector.edge_head.parameters())),
                    "T":(terms.transition,list(self.selector.t_interaction.parameters()))}
            measured={}
            for name,(loss,params) in groups.items():
                grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                norms=[g.detach().float().square().sum() for g in grads if g is not None]
                measured[name+"_loss"]=float(loss.detach())
                measured[name+"_gradient_norm"]=float(torch.stack(norms).sum().sqrt()) if norms else 0.
                if name=="E":
                    measured["E_nonzero_parameter_tensors"]=sum(int(g is not None and bool(g.detach().ne(0).any())) for g in grads)
                    readout_ids = {id(p) for p in self.selector.e_interaction.readout.parameters()}
                    for part, selected in (("readout", True), ("encoder", False)):
                        selected_grads = [g for p,g in zip(params,grads)
                                          if (id(p) in readout_ids) == selected and g is not None]
                        pieces = [g.detach().float().square().sum() for g in selected_grads]
                        measured["E_"+part+"_gradient_norm"] = float(torch.stack(pieces).sum().sqrt()) if pieces else 0.
                        measured["E_"+part+"_nonzero_parameter_tensors"] = sum(int(bool(g.detach().ne(0).any())) for g in selected_grads)
            measured.update(epoch=int(epoch),arm=self.cf_arm,training=True,
                            active_queries=float(probe.get("active_queries",clean.new_zeros(()))))
            self.cf_last_gradient = measured
            if self.cf_probe_dir:
                path=Path(self.cf_probe_dir);path.mkdir(parents=True,exist_ok=True)
                (path/("epoch_%03d.json"%epoch)).write_text(json.dumps(measured,indent=2))
        self.r50_last_terms=None
        self.cf_last_evidence=None
        return result

    def runtime_probe(self):
        return {"contract": self.experiment_contract(), "arm": self.cf_arm,
                "E": dict(self.cf_last_probe), "gradient": dict(self.cf_last_gradient)}

    def r50_adjust_terms(self,terms,outputs,batch,epoch):
        result=super().r50_adjust_terms(terms,outputs,batch,epoch)
        metrics=dict(result.metrics)
        for key in list(metrics):
            if key.startswith("local_evidence/"):
                metrics["e_counterfactual/"+key[len("local_evidence/"):]]=metrics.pop(key)
        metrics["term_source/evidence_local_ordinal"]=0.
        metrics["term_source/evidence_counterfactual_local"]=1.
        return dataclasses.replace(result,metrics=metrics)

    def experiment_contract(self):
        c=dict(super().experiment_contract())
        c["e_counterfactual"]={"arm":self.cf_arm,"E_weight":.1,"margin":.2,"temperature":.2,
           "saliency_used_by_E":False,"aux_dropout":"disabled_all_four_arms",
           "main_dropout":.5,"new_parameters":False,"checkpoint_loads_in_factory":0,
           "weak_negative_policy":"reciprocal_different_video_query_jaccard_lt_0.5",
           "GT_pool":"per_GT_overlap_normalize_then_equal_GT_per_query",
           "gt_coordinate":"norm_times_FPN_duration_over_2_real_clip_mask",
           "time_axis_scope":"QV_2sec_annotation_domain_separate_actual_feature_cap75_FPN4",
           "prelaunch_feature_duration_audit_required":True}
        return c

class CounterfactualHD(CounterfactualMixin,joint_model.EventFieldNet):pass
class CounterfactualMR(CounterfactualMixin,MomentModel):pass

def build_model(config,*,cf_arm="C02",annotation_files=None,enable_highlight=True,**kwargs):
    if cf_arm != "C02":raise ValueError("This release implements C02 only")
    if not annotation_files:raise ValueError("Need dataset time-axis annotations")
    model=joint_model.build_model(config,annotation_files=annotation_files,enable_highlight=enable_highlight,**kwargs)
    model.__class__=CounterfactualHD if enable_highlight else CounterfactualMR
    model.cf_arm=cf_arm;model.cf_clip_counts={};model.cf_probe_epoch=None;model.cf_probe_dir=None
    model.cf_last_evidence=None;model.cf_last_probe={};model.cf_last_gradient={};model.r50_capture_terms=True
    model.cf_clip_counts=annotation_clip_counts(annotation_files)
    return model
