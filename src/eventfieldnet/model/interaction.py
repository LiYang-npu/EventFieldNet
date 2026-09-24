from ._rng import install_cpu_generator_state

"""Round31 trainable raw V/T interactions and label-only local E eligibility."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RawInteraction(nn.Module):
    def __init__(
        self,
        seed,
        transition=False,
        projected_cosine_attention=False,
        depth=1,
        feedforward=False,
        dropout=0.0,
    ):
        super().__init__()
        self.projected_cosine_attention = bool(projected_cosine_attention)
        self.depth = int(depth)
        self.feedforward = bool(feedforward)
        self.dropout = nn.Dropout(dropout)
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        with torch.random.fork_rng(devices=[]):
            install_cpu_generator_state(seed)
            self.video = nn.Linear(512, 64, bias=False)
            self.key = nn.Linear(512, 64, bias=False)
            self.value = nn.Linear(512, 64, bias=False)
            self.readout = nn.Linear(64, 1, bias=False)
            nn.init.zeros_(self.readout.weight)
            if transition:
                self.end_readout = nn.Linear(64, 1, bias=False)
                nn.init.zeros_(self.end_readout.weight)
            # round32 MR57 structural ablation: depth/feedforward extension.
            # Constructed AFTER all original params so the CPU RNG draws
            # above are byte-identical to depth=1/feedforward=False regardless
            # of these settings; new params are only created (and only draw
            # from the generator) when actually requested.
            if self.depth > 1:
                self.refine_key = nn.ModuleList(
                    [nn.Linear(512, 64, bias=False) for _ in range(self.depth - 1)]
                )
                self.refine_value = nn.ModuleList(
                    [nn.Linear(512, 64, bias=False) for _ in range(self.depth - 1)]
                )
            if self.feedforward:
                self.ffn = nn.Sequential(
                    nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64)
                )
                nn.init.zeros_(self.ffn[-1].weight)
                nn.init.zeros_(self.ffn[-1].bias)

    def encode(self, video, text, video_pad, text_pad):
        if video.shape[-1] != 512 or text.shape[-1] != 512:
            raise ValueError("raw video/text must be512")
        if text_pad is None:
            text_pad = torch.zeros(text.shape[:2], dtype=torch.bool, device=text.device)
        if (~text_pad.bool()).sum(1).eq(0).any():
            raise ValueError("empty valid query token set")
        # Explicit float32 even under training autocast, matching field arithmetic.
        with torch.autocast(device_type=video.device.type, enabled=False):
            v = F.normalize(video.float(), dim=-1, eps=1e-6)
            t = F.normalize(text.float(), dim=-1, eps=1e-6)
            q = self.video(v)
            k = self.key(t)
            u = self.value(t)
            a = (
                self.attention_logits(q, k)
                .masked_fill(text_pad[:, None, :].bool(), float("-inf"))
                .softmax(-1)
            )
            h = F.gelu(q) * (a @ u)
            for i in range(self.depth - 1):
                k2 = self.refine_key[i](t)
                u2 = self.refine_value[i](t)
                a2 = (
                    self.attention_logits(h, k2)
                    .masked_fill(text_pad[:, None, :].bool(), float("-inf"))
                    .softmax(-1)
                )
                h = h + F.gelu(h) * (a2 @ u2)
            if self.feedforward:
                h = h + self.ffn(h)
            h = self.dropout(h)
            return h.masked_fill(video_pad[..., None].bool(), 0)

    def attention_logits(self, q, k):
        if self.projected_cosine_attention:
            return F.normalize(q, dim=-1, eps=1e-6) @ F.normalize(
                k, dim=-1, eps=1e-6
            ).transpose(-1, -2)
        return q @ k.transpose(-1, -2) / math.sqrt(64)

    def encode_difference(self, difference, text, pad, text_pad):
        with torch.autocast(device_type=difference.device.type, enabled=False):
            # Difference of unit raw videos, deliberately not normalized again.
            q = self.video(difference.float())
            t = F.normalize(text.float(), dim=-1, eps=1e-6)
            k = self.key(t)
            u = self.value(t)
            a = (
                self.attention_logits(q, k)
                .masked_fill(text_pad[:, None, :].bool(), float("-inf"))
                .softmax(-1)
            )
            return (F.gelu(q) * (a @ u)).masked_fill(pad[..., None].bool(), 0)

    @staticmethod
    def differences(z, pad):
        left = torch.cat((z[:, :1], z[:, :-1]), 1)
        right = torch.cat((z[:, 1:], z[:, -1:]), 1)
        lp = torch.cat((pad[:, :1], pad[:, :-1]), 1)
        rp = torch.cat((pad[:, 1:], pad[:, -1:]), 1)
        # Replicate current token at unavailable neighbor, never difference into padding.
        return (z - left).masked_fill((pad | lp)[..., None], 0), (
            z - right
        ).masked_fill((pad | rp)[..., None], 0)


def local_e_mask(outputs, batch, g):
    from field_core.adapter import _batch_parts, _metadata_rows
    from .losses import _target_spans, _pad_empty_gt

    inp, targets, meta = _batch_parts(batch)
    B, L, _ = g.valid.shape
    result = torch.zeros_like(g.valid)
    sal = targets.get("saliency_all_labels")
    if sal is None:
        return result
    sal = sal.detach().cpu()
    pad = inp["video_padding_mask"].detach().cpu()
    rows = _metadata_rows(meta, B)
    spans, gm = _target_spans(outputs, _pad_empty_gt(batch))
    spans = spans.detach().cpu()
    gm = gm.cpu()
    step = g.one_step.cpu()
    valid = g.valid.cpu()
    idx = torch.arange(L)
    starts = idx[:, None]
    ends = idx[None, :] + 1
    width = ends - starts
    for b, row in enumerate(rows):
        rated = torch.zeros(L, dtype=torch.bool)
        for x in (row or {}).get("relevant_clip_ids", []):
            if int(x) != x or not 0 <= int(x) < L:
                raise ValueError("invalid explicitly rated clip ID")
            rated[int(x)] = True
        if sal.shape != (B, L) or not torch.isfinite(sal[b, rated]).all():
            raise ValueError("invalid saliency")
        for gi in gm[b].nonzero().flatten().tolist():
            # Compare in original normalized units, never round GT boundaries.
            inside = (
                (idx * step[b] >= spans[b, gi].min())
                & ((idx + 1) * step[b] <= spans[b, gi].max())
                & ~pad[b]
            )
            anchors = (inside & rated & sal[b].gt(0)).nonzero().flatten()
            if anchors.numel() == 0:
                continue
            anchor = int(anchors[sal[b, anchors].argmax()])
            n = int(inside.sum())
            count = torch.cat(
                (torch.zeros(1, dtype=torch.long), inside.long().cumsum(0))
            )
            contained = count[ends] - count[starts] == width
            result[b] = (
                (
                    valid[b]
                    & contained
                    & (2 * width <= n)
                    & (starts <= anchor)
                    & (ends > anchor)
                )
                | result[b].cpu()
            ).to(result.device)
    return result


def interaction_probe(model, outputs, batch, terms, context):
    from field_core.adapter import (
        _batch_parts,
        _valid_wrong_query_pairs,
        _metadata_rows,
    )

    inp, _, meta = _batch_parts(batch)
    pad = inp["video_padding_mask"].bool()
    f = outputs.trifield_output
    valid = f.valid
    rows = _metadata_rows(meta, valid.shape[0])
    _, valid_pair = _valid_wrong_query_pairs(meta, valid.shape[0])
    valid_pair = valid_pair.to(valid.device)
    wrong_token_mask = (~pad) & valid_pair[:, None]
    token_ids = []
    for b in range(len(pad)):
        ids = (~pad[b]).nonzero().flatten()
        # Equal per-query count using the training minimum (bounded at4), shared all fields.
        token_ids.append(ids)
    count = min([4] + [len(x) for x in token_ids])
    token_ids = [
        x[torch.linspace(0, len(x) - 1, count, device=x.device).round().long()]
        if count
        else x[:0]
        for x in token_ids
    ]
    result = {
        "scope": "actual deployed E/T; existing correct/wrong/mask forwards only",
        "enabled_E": model.selector.learned_e,
        "enabled_T": model.selector.learned_t,
        "sample_per_query": count,
        "sample_qids": [str((r or {}).get("qid")) for r in rows],
        "sample_token_ids": [x.tolist() for x in token_ids],
    }

    def stats(z):
        sample = (
            torch.cat([z[b, ids] for b, ids in enumerate(token_ids)], 0)
            .detach()
            .float()
        )
        if not sample.numel():
            return {
                "sample_count": 0,
                "finite": True,
                "variance": 0.0,
                "effective_rank": 0.0,
                "zero_variance": True,
            }
        center = sample - sample.mean(0, keepdim=True)
        power = torch.linalg.svdvals(center).square()
        mass = power / power.sum().clamp_min(1e-30)
        return {
            "sample_count": len(sample),
            "finite": bool(torch.isfinite(sample).all()),
            "variance": float(center.square().mean()),
            "effective_rank": float(
                torch.exp(-(mass * mass.clamp_min(1e-30).log()).sum())
            )
            if power.sum() > 0
            else 0.0,
            "zero_variance": bool(center.square().sum() == 0),
        }

    state = outputs.extension_output.state
    for i, name in enumerate(("evidence", "support", "transition")):
        result["original_" + name] = stats(state.role_updates[:, :, i])
    for branch in ("e", "t"):
        z = getattr(f, "round31_z_" + branch, None)
        if z is None:
            continue
        result[branch] = stats(z)
        ablated = model.selector.score_without_interaction(f, branch)
        result[branch]["ablation_F_mean_abs"] = (
            float((f.score - ablated)[valid].detach().abs().mean())
            if valid.any()
            else None
        )
        if branch == "e":
            from field_core.selector import centered_bounded

            only_new = centered_bounded(f.round31_delta_e, valid)
            old_removed = model.selector.compose_score(f, {"evidence": only_new})
            residual = f.round31_delta_e[valid]
        else:
            old_removed = model.selector.compose_score(
                f,
                {
                    "transition_start": torch.tanh(f.round31_delta_ts).masked_fill(
                        ~valid, 0
                    ),
                    "transition_end": torch.tanh(f.round31_delta_te).masked_fill(
                        ~valid, 0
                    ),
                },
            )
            residual = torch.cat((f.round31_delta_ts[valid], f.round31_delta_te[valid]))
        result[branch]["old_path_removed_F_mean_abs"] = (
            float((f.score - old_removed)[valid].detach().abs().mean())
            if valid.any()
            else None
        )
        result[branch]["raw_residual_mean_abs"] = (
            float(residual.detach().abs().mean()) if residual.numel() else None
        )
        result[branch]["raw_residual_finite"] = bool(torch.isfinite(residual).all())
        wrong = context.get("wrong_field")
        wz = getattr(wrong, "round31_z_" + branch, None)
        if wz is not None:
            result[branch]["wrong_query_z_mean_abs"] = (
                float((z - wz)[wrong_token_mask].detach().abs().mean())
                if wrong_token_mask.any()
                else None
            )
            result[branch]["wrong_query_valid_token_count"] = int(
                wrong_token_mask.sum()
            )
            name = "evidence" if branch == "e" else "transition_start"
            result[branch]["wrong_query_deployed_mean_abs"] = (
                float(
                    (getattr(f, name) - getattr(wrong, name))[valid]
                    .detach()
                    .abs()
                    .mean()
                )
                if valid.any()
                else None
            )
        if branch == "t":
            dl, dr = f.round31_t_left, f.round31_t_right
            lp = torch.cat((pad[:, :1], pad[:, :-1]), 1)
            rp = torch.cat((pad[:, 1:], pad[:, -1:]), 1)
            result[branch]["padding_edges_zero"] = bool(
                dl[pad | lp].eq(0).all() and dr[pad | rp].eq(0).all()
            )
    local = local_e_mask(outputs, batch, terms.geometry)
    old = terms.geometry.valid & terms.geometry.max_iou.ge(0.7)
    _, pair = _valid_wrong_query_pairs(meta, valid.shape[0])
    pair = pair.to(valid.device)[:, None, None]
    old &= pair
    local &= pair
    potential = local & ~old
    union = old | local if context.get("round31_bits", 0) & 1 else old
    added = union & ~old
    den = union.sum().clamp_min(1)
    dedup = bool((old & added).sum() == 0 and union.sum() == old.sum() + added.sum())
    assert dedup
    result["eligibility"] = {
        "enabled": bool(context.get("round31_bits", 0) & 1),
        "old_counts": old.flatten(1).sum(1).tolist(),
        "added_counts": added.flatten(1).sum(1).tolist(),
        "potential_added_counts": potential.flatten(1).sum(1).tolist(),
        "union_counts": union.flatten(1).sum(1).tolist(),
        "active_queries": int(union.flatten(1).any(1).sum()),
        "old_weight_mass": float(old.sum() / den),
        "added_weight_mass": float(added.sum() / den),
        "duplicate_free": dedup,
        "actual_E_loss": float(terms.evidence.detach()),
    }
    wrong = context.get("wrong_field")
    if wrong is not None:
        gap = (f.evidence - wrong.evidence).detach()
        L = valid.shape[-1]
        ix = torch.arange(L, device=valid.device)
        width = (ix[None, :] - ix[:, None] + 1).expand_as(valid)
        for name, mask in [
            ("old", old),
            ("potential_added", potential),
            ("potential_iou_lt03", potential & terms.geometry.max_iou.lt(0.3)),
            ("potential_iou_ge03", potential & terms.geometry.max_iou.ge(0.3)),
            ("potential_width1", potential & width.eq(1)),
            ("potential_widthgt1", potential & width.gt(1)),
        ]:
            x = gap[mask]
            result["eligibility"][name + "_margin"] = {
                "count": len(x),
                "mean": float(x.mean()) if len(x) else None,
                "satisfied_fraction": float(x.ge(0.2).float().mean())
                if len(x)
                else None,
            }
    pairs = []
    for b, g, typ, a, c in context["support_pairs"]:
        pairs.append(
            {
                name: float(
                    (
                        getattr(f, name)[b].flatten()[a]
                        - getattr(f, name)[b].flatten()[c]
                    ).detach()
                )
                for name in ("evidence", "support", "score")
            }
        )
    result["S_joint_actual_pairs"] = {
        "count": len(pairs),
        "rows": pairs,
        "scope": "same supervised pairs; E/S/F differences, not official MR",
    }
    directions = [r for r in context.get("direction_measurements", []) if r["new"]]
    if directions:
        dc = torch.stack([r["correct"].detach() for r in directions])
        dw = torch.stack([r["wrong"].detach() for r in directions])
        success = (dc - dw).ge(0.2)
        result["T_direction"] = {
            "count": len(directions),
            "correct_positive_count": int(dc.gt(0).sum()),
            "correct_negative_count": int(dc.lt(0).sum()),
            "wrong_negative_count": int(dw.lt(0).sum()),
            "query_margin_success_count": int(success.sum()),
            "success_correct_nonpositive_count": int((success & dc.le(0)).sum()),
            "success_correct_nonpositive_fraction": float(
                (success & dc.le(0)).sum() / success.sum()
            )
            if success.any()
            else None,
        }
    else:
        result["T_direction"] = {"count": 0}
    reps = {
        "old_E": state.role_updates[:, :, 0],
        "old_S": state.role_updates[:, :, 1],
        "old_T": state.role_updates[:, :, 2],
    }
    for name in ("e", "t"):
        z = getattr(f, "round31_z_" + name, None)
        if z is not None:
            reps["new_" + name] = z
    grams = {}
    for name, z in reps.items():
        x = (
            torch.cat([z[b, ids] for b, ids in enumerate(token_ids)], 0)
            .detach()
            .float()
        )
        x = x - x.mean(0, keepdim=True) if len(x) else x
        grams[name] = x @ x.T
    result["cross_field_centered_gram_cosine"] = {}
    names = list(grams)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            den = grams[a].norm() * grams[b].norm()
            result["cross_field_centered_gram_cosine"][a + "__" + b] = (
                float((grams[a] * grams[b]).sum() / den) if den > 1e-20 else None
            )
    result["cross_field_scope"] = (
        "same fixed query-balanced tokens; centered Gram cosine across feature spaces, not semantic correctness"
    )
    from .mechanism_probe import mechanism_probe

    result["mechanisms"] = mechanism_probe(model, outputs, context, valid_pair)
    return result
