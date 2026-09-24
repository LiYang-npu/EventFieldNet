import torch
from torch.nn import functional as F
from .interaction import RawInteraction


@torch.no_grad()
def mechanism_probe(model, outputs, context, valid_pair):
    f = outputs.trifield_output
    s = model.selector
    valid = f.valid
    v, t, pad, tpad = outputs.extension_output.state.round31_raw_inputs
    result = {
        "fixed_control_bits": 7,
        "control_scope": "historical E/T controls, not X experiment index",
        "projected_cosine_attention": s.projected_cosine_attention,
        "local_top_quarter_e": s.local_top_quarter_e,
        "difference_before_interaction_t": s.difference_before_interaction_t,
        "attention": {},
    }
    for name in ("e", "t"):
        layer = getattr(s, name + "_interaction")
        vn = F.normalize(v.float(), dim=-1, eps=1e-6)
        pm = pad
        if name == "t" and s.difference_before_interaction_t:
            dl, dr = RawInteraction.differences(vn, pad)
            vn = torch.cat((dl, dr), 1)
            lp = torch.cat((pad[:, :1], pad[:, :-1]), 1)
            rp = torch.cat((pad[:, 1:], pad[:, -1:]), 1)
            pm = torch.cat((pad | lp, pad | rp), 1)
        q = layer.video(vn)
        k = layer.key(F.normalize(t.float(), dim=-1, eps=1e-6))
        a = (
            layer.attention_logits(q, k)
            .masked_fill(tpad[:, None, :], float("-inf"))
            .softmax(-1)
        )
        n = (~tpad).sum(1)
        mask = (~pm) & n[:, None].gt(1)
        entropy = (
            -(a * a.clamp_min(1e-30).log()).sum(-1)
            / n.clamp_min(2).float().log()[:, None]
        )
        l1 = (a - (~tpad).float()[:, None, :] / n[:, None, None]).abs().sum(-1)
        result["attention"][name] = {
            "entropy_normalized_mean": float(entropy[mask].mean())
            if mask.any()
            else None,
            "l1_to_uniform_mean": float(l1[mask].mean()) if mask.any() else None,
            "finite": bool(torch.isfinite(a).all()),
            "valid_multitoken_positions": int(mask.sum()),
        }
    delta = f.round31_delta_e
    count = valid.sum((-1, -2), keepdim=True).clamp_min(1)
    dc = delta - (delta * valid).sum((-1, -2), keepdim=True) / count
    raw = f.raw_evidence
    center = raw - (raw * valid).sum((-1, -2), keepdim=True) / count
    result["E_stages"] = {
        "token_abs_mean": float(f.round31_e_token[~pad].abs().mean()),
        "span_abs_mean": float(delta[valid].abs().mean()),
        "centered_abs_mean": float(dc[valid].abs().mean()),
        "tanh_derivative_mean": float((1 - torch.tanh(center[valid]).square()).mean()),
        "finite": bool(torch.isfinite(dc[valid]).all()),
        "length_bins": {},
    }
    L = valid.shape[-1]
    ix = torch.arange(L, device=valid.device)
    width = ix[None, :] - ix[:, None] + 1
    for name, mask in [
        ("width1", valid & width.eq(1)),
        ("width2to8", valid & width.ge(2) & width.le(8)),
        ("width_gt8", valid & width.gt(8)),
    ]:
        result["E_stages"]["length_bins"][name] = {
            "count": int(mask.sum()),
            "centered_abs_mean": float(dc[mask].abs().mean()) if mask.any() else None,
        }
    lp = torch.cat((pad[:, :1], pad[:, :-1]), 1)
    rp = torch.cat((pad[:, 1:], pad[:, -1:]), 1)
    result["T_padding_edges_zero"] = bool(
        f.round31_t_left[pad | lp].eq(0).all()
        and f.round31_t_right[pad | rp].eq(0).all()
    )
    wrong = context.get("wrong_field")
    mask = valid & valid_pair[:, None, None]
    result["valid_wrong_pair_queries"] = int(valid_pair.sum())
    result["T_new_query_difference_abs_mean"] = None
    if wrong is not None and mask.any():
        c = f.score - s.score_without_interaction(f, "t")
        w = wrong.score - s.score_without_interaction(wrong, "t")
        result["T_new_query_difference_abs_mean"] = float((c - w)[mask].abs().mean())
    return result
