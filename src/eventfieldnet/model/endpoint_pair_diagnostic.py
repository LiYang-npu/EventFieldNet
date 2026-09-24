"""CPU-only, standard-library diagnostic; never loads a model or calls an evaluator.

Input JSON: {metadata: {checkpoint_sha256, seed, epoch, panel_qid_sha256,
coordinate_system, transition_coefficients: [0.5,0.5]}, queries: [...]}
Each query: qid, candidate_start/end [C], valid [C], score [C],
transition_start/end [C] (activated fields, NOT raw logits), gt_spans [G,2],
gt_mask [G], one_step > 0. Optional gt_duration_seconds [G] enables <=10s scope.
Coordinates and one_step MUST come from training candidate_geometry; GT from
prepared training targets. This is training geometry, not official seconds/AP.
All arrays must be detached CPU lists. Invalid/padded candidates are excluded.
CLI: python endpoint_pair_diagnostic.py --self-test
     python endpoint_pair_diagnostic.py --input cache.json --output report.json
"""

import argparse
import json
import math
from pathlib import Path


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = a[1] - a[0] + b[1] - b[0] - inter
    return inter / union if union > 0 else 0.0


def endpoint_quality(span, gt, step):
    lo, hi = sorted(gt)
    den = max(hi - lo, 1.0e-6, step)  # identical training width clamp
    return tuple(
        max(0.0, min(1.0, 1.0 - abs(x - y) / den)) for x, y in zip(span, (lo, hi))
    )


def pair_semantics(span, gts, step, tol=1.0e-6):
    values = [endpoint_quality(span, gt, step) for gt in gts]
    if not values:
        return {"active": False, "strong_cross": False, "delta_pair": 0.0}
    us, ue = map(max, zip(*values))
    training = any(us - s <= tol and ue - e <= tol for s, e in values)
    quality = max(iou(span, sorted(gt)) for gt in gts)
    return dict(
        active=True,
        start_label=2 * us - 1,
        end_label=2 * ue - 1,
        start_quality=us,
        end_quality=ue,
        delta_pair=(us + ue - max(s + e for s, e in values)) / 2,
        tie_aware_conflict=not training,
        legacy_argmax_conflict=(
            max(range(len(values)), key=lambda i: values[i][0])
            != max(range(len(values)), key=lambda i: values[i][1])
        ),
        max_iou=quality,
        strong_cross=(not training and us >= 0.7 and ue >= 0.7 and quality < 0.7),
    )


def matching(ids, overlaps, gtids, threshold):
    # Maximum cardinality one-to-one matching, not score-greedy matching.
    assigned = {}

    def visit(c, seen):
        for g in gtids:
            if overlaps[c][g] < threshold or g in seen:
                continue
            seen.add(g)
            if g not in assigned or visit(assigned[g], seen):
                assigned[g] = c
                return True
        return False

    for c in ids:
        visit(c, set())
    return len(assigned)


def diagnose_query(q):
    keys = (
        "candidate_start",
        "candidate_end",
        "valid",
        "score",
        "transition_start",
        "transition_end",
    )
    n = len(q["score"])
    if n > 65536 or any(len(q[k]) != n for k in keys):
        raise ValueError("candidate lengths differ or exceed CPU bound")
    step = q["one_step"]
    if not math.isfinite(step) or step <= 0:
        raise ValueError("one_step must be positive and finite")
    if len(q["gt_mask"]) != len(q["gt_spans"]):
        raise ValueError("GT mask mismatch")
    gtids = [g for g, mask in enumerate(q["gt_mask"]) if mask]
    gts = [q["gt_spans"][g] for g in gtids]
    if len(gts) > 128:
        raise ValueError("GT count exceeds CPU bound")
    ids = [i for i in range(n) if q["valid"][i]]
    for i in ids:
        if not all(math.isfinite(q[k][i]) for k in keys if k != "valid"):
            raise ValueError("nonfinite valid candidate")
        if q["candidate_end"][i] < q["candidate_start"][i]:
            raise ValueError("reversed candidate")
    if any(not all(math.isfinite(x) for x in gt) for gt in gts):
        raise ValueError("nonfinite GT")
    spans = {i: (q["candidate_start"][i], q["candidate_end"][i]) for i in ids}
    sem = {i: pair_semantics(spans[i], gts, step) for i in ids}
    overlaps = {i: [iou(spans[i], sorted(gt)) for gt in gts] for i in ids}
    order = sorted(ids, key=lambda i: (-q["score"][i], i))
    nms = []
    for i in order:
        if all(iou(spans[i], spans[j]) <= 0.5 for j in nms):
            nms.append(i)
        if len(nms) == 30:
            break
    # Exact model formula; subtract bounded, coefficient-weighted T only.
    t = {i: 0.5 * (q["transition_start"][i] + q["transition_end"][i]) for i in ids}
    no_t = {
        i: q["without_transition_score"][i]
        if "without_transition_score" in q
        else q["score"][i] - t[i]
        for i in ids
    }
    scopes = {"all_gt": list(range(len(gts)))}
    if "gt_duration_seconds" in q:
        scopes["short_le10s"] = [
            j for j, g in enumerate(gtids) if q["gt_duration_seconds"][g] <= 10
        ]
    result = {
        "qid": q["qid"],
        "gt_count": len(gts),
        "multi_gt": len(gts) > 1,
        "valid_candidates": len(ids),
        "selections": {},
        "ranking": {},
    }
    for name, chosen in [
        ("legal", ids),
        ("raw10", order[:10]),
        ("raw30", order[:30]),
        ("nms30", nms),
    ]:
        cross = [i for i in chosen if sem[i]["strong_cross"]]
        result["selections"][name] = {
            "count": len(chosen),
            "strong_cross_count": len(cross),
            "strong_cross_rate": len(cross) / len(chosen) if chosen else None,
            "tie_aware_conflict_count": sum(
                sem[i].get("tie_aware_conflict", False) for i in chosen
            ),
            "legacy_conflict_count": sum(
                sem[i].get("legacy_argmax_conflict", False) for i in chosen
            ),
            "delta_pair_mean": sum(sem[i]["delta_pair"] for i in chosen) / len(chosen)
            if chosen
            else None,
            "strong_cross_T_mean": sum(t[i] for i in cross) / len(cross)
            if cross
            else None,
        }
        if name != "legal":
            result["selections"][name]["matching"] = {
                scope: {
                    str(th): {
                        "gt_count": len(gs),
                        "hits": matching(chosen, overlaps, gs, th),
                    }
                    for th in (0.7, 0.75)
                }
                for scope, gs in scopes.items()
            }
    for th in (0.7, 0.75):
        rows = []
        for g in range(len(gts)):
            good = [i for i in ids if overlaps[i][g] >= th]
            if not good:
                rows.append({"gt_index": gtids[g], "reachable": False})
                continue
            a = max(good, key=lambda i: (q["score"][i], -i))
            bad = [i for i in ids if sem[i]["strong_cross"]]
            ahead = [i for i in bad if q["score"][i] > q["score"][a]]
            flips = [i for i in ahead if no_t[i] < no_t[a]]
            rows.append(
                {
                    "gt_index": gtids[g],
                    "reachable": True,
                    "best_good_index": a,
                    "cross_ahead_count": len(ahead),
                    "T_flip_count": len(flips),
                    "raw30_T_flip_count": sum(i in order[:30] for i in flips),
                    "nms30_T_flip_count": sum(i in nms for i in flips),
                }
            )
        result["ranking"][str(th)] = rows
    return result


def self_test():
    cross = pair_semantics((0, 12), [(0, 2), (10, 12)], 1)
    assert cross["strong_cross"] and cross["delta_pair"] == 0.5
    tied = pair_semantics((0, 2), [(0, 2), (0, 2)], 1)
    assert not tied["tie_aware_conflict"] and tied["delta_pair"] == 0
    zero = pair_semantics((30, 32), [(0, 2), (10, 12)], 1)
    assert not zero["strong_cross"] and zero["start_label"] == -1
    single = pair_semantics((0, 2), [(0, 2)], 1)
    assert not single["tie_aware_conflict"] and single["start_label"] == 1
    assert not pair_semantics((0, 2), [], 1)["active"]
    # Exact .5 weighting: b overtakes a only via T; no-T removes one, not two.
    q = dict(
        qid="synthetic",
        candidate_start=[0, 0],
        candidate_end=[2, 12],
        valid=[True, True],
        score=[1.0, 1.5],
        transition_start=[0.0, 1.0],
        transition_end=[0.0, 1.0],
        gt_spans=[[0, 2], [10, 12]],
        gt_mask=[True, True],
        one_step=1.0,
    )
    r = diagnose_query(q)
    assert r["ranking"]["0.7"][0]["T_flip_count"] == 1
    assert r["selections"]["raw30"]["strong_cross_T_mean"] == 1.0
    assert matching([0, 1], {0: [1, 1], 1: [1, 0]}, [0, 1], 0.7) == 2
    # A tied end maximum shared with start maximum is NOT a true conflict.
    v = pair_semantics((10, 30), [(0, 2), (10, 12)], 1)
    assert v["legacy_argmax_conflict"] and not v["tie_aware_conflict"]
    return {
        "passed": 8,
        "device": "cpu",
        "dependencies": "Python standard library",
        "checks": [
            "multi_gt",
            "duplicate_tie",
            "zero_quality",
            "single_gt",
            "empty_gt",
            "actual_half_coefficient_flip",
            "maximum_matching",
            "partial_tie",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--input")
    p.add_argument("--output")
    args = p.parse_args()
    if args.self_test:
        result = self_test()
    else:
        if not args.input:
            p.error("--input or --self-test required")
        data = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
        meta = data["metadata"]
        for key in (
            "checkpoint_sha256",
            "seed",
            "epoch",
            "panel_qid_sha256",
            "coordinate_system",
        ):
            if key not in meta:
                raise ValueError("missing metadata " + key)
        if meta.get("transition_coefficients") != [0.5, 0.5]:
            raise ValueError("this probe requires verified Round5 .5/.5 T coefficients")
        if not 0 < len(data["queries"]) <= 256:
            raise ValueError("panel bound: 1..256 queries")
        result = {
            "schema": "endpoint_pair_cpu_v1",
            "metadata": meta,
            "official_validation": False,
            "selection": "stable score order; hard NMS suppress IoU>.5; GT never selects predictions",
            "queries": [diagnose_query(q) for q in data["queries"]],
        }
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(
        output
        if args.self_test
        else json.dumps({"queries": len(result["queries"]), "output": args.output})
    )


if __name__ == "__main__":
    main()
