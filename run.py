"""Portable entry points for the unchanged V00 reference implementation."""

from pathlib import Path
import argparse, json, os, sys, hashlib

ROOT = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def materialize(seed, output):
    if seed not in range(2041, 2046):
        raise ValueError(
            "This release includes probe panels for the five reported seeds2041-2045"
        )
    template = ROOT / "configs" / "qvhighlights.json"
    if not template.exists():
        raise ValueError("Use one of the five preregistered seeds: 2041..2045")
    text = template.read_text().replace("@ROOT@", ROOT.as_posix())
    raw = json.loads(text)
    raw["output_dir"] = str(Path(output).resolve())
    raw["seed"] = seed
    raw["name"] = f"eventfieldnet_seed{seed}"
    for p in reversed(raw["source_roots"]):
        sys.path.insert(0, p)
    return raw


def save_config(raw, *, create_output=True):
    out = Path(raw["output_dir"])
    if out.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {out}")
    if create_output:
        out.mkdir(parents=True, exist_ok=False)
        cfg = out / "config.json"
    else:
        # Formal runner owns its output directory and rejects any preexisting content.
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg = out.parent / (out.name + ".config.json")
    with cfg.open("x") as f:
        json.dump(raw, f, indent=2)
    return cfg


def check_data():
    base = ROOT / "data/qvhighlights"
    for relative in [
        "annotation/highlight_train_release.jsonl",
        "annotation/highlight_val_release.jsonl",
        "custom_features/video",
        "custom_features/custom_text",
    ]:
        if not (base / relative).exists():
            raise FileNotFoundError(f"Missing {base / relative}; see DATA.md")
    expected = json.loads((ROOT / "docs/annotation_sha256.json").read_text())
    for name, digest in expected.items():
        if (
            hashlib.sha256((base / "annotation" / name).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError(f"Annotation does not match the reference release: {name}")


def normalize_factory(factory):
    if factory["target"].startswith("eventfieldnet."):
        return factory
    import re

    mapping = json.loads((ROOT / "docs/import_compatibility.json").read_text())
    pattern = re.compile(
        r"\b("
        + "|".join(map(re.escape, sorted(mapping, key=len, reverse=True)))
        + r")\b"
    )
    return json.loads(pattern.sub(lambda m: mapping[m.group()], json.dumps(factory)))


def evaluate_checkpoint(checkpoint, raw, output, omit=None, split="val"):
    import torch
    from eventfieldnet import precision_evaluation as cli

    # Relocate only paths; reject using this entry point for another recipe/seed.
    payload = torch.load(checkpoint, map_location="cpu")
    old = payload["config"]
    assert old["seed"] == raw["seed"]
    assert normalize_factory(old["model_factory"]) == raw["model_factory"], (
        "Wrong model recipe"
    )
    cfg = json.loads(json.dumps(raw))
    if omit:
        cfg["model_factory"]["target"] = "eventfieldnet.field_ablation:build_model"
        cfg["model_factory"]["kwargs"]["audit_omit"] = omit
    if split == "test":
        annotation = ROOT / "data/qvhighlights/annotation/highlight_test_with_gt.jsonl"
        if not annotation.is_file():
            raise FileNotFoundError(
                "Provide the official test annotation file; test is never used for model selection"
            )
        if (
            hashlib.sha256(annotation.read_bytes()).hexdigest()
            != "bd50ec6bb5dd3f72571126ba5fdc7418efdd9219a9487984e483a02e2ce5d493"
        ):
            raise ValueError(
                "Test annotation hash differs from the documented official release"
            )
        cfg["data_factory"] = {
            "target": "eventfieldnet.heldout_data:build",
            "kwargs": {
                "root": str(ROOT),
                "annotation": str(annotation),
                "annotation_sha256": hashlib.sha256(
                    annotation.read_bytes()
                ).hexdigest(),
            },
        }
    previous = cli.original._checkpoint_config
    cli.original._checkpoint_config = lambda payload, config_path: cfg
    try:
        return cli.evaluate_checkpoint(
            checkpoint=Path(checkpoint),
            output_dir=Path(output),
            root=ROOT,
            project_root=ROOT,
            precision="fp32",
        )
    finally:
        cli.original._checkpoint_config = previous


def smoke(raw, cfg):
    """Four real batches; diagnostic weights are never used for formal runs."""
    import torch, math

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from eventfieldnet.runtime import runner
    from eventfieldnet.train import install_runtime
    from training.runner import set_model_epoch

    rt, _, _, _ = runner._build_phase_runtime(cfg, raw)
    install_runtime(rt, raw)
    m = rt.model
    m.train()
    m.r50_capture_terms = True
    params = [p for p in m.parameters() if p.requires_grad]
    records = []
    for i, data in enumerate(rt.bundle.train_loader):
        if i == 4:
            break
        b = rt.bundle.prepare_batch(data, rt.device)
        rt.optimizer.zero_grad(set_to_none=True)
        set_model_epoch(m, 1, True, set(rt.registered_trainable_ids))
        pred = m(b.inputs)
        loss = m.compute_loss(pred, b, None, 1)
        terms = m.r50_last_terms
        norms = {}
        for key in ("evidence", "support", "transition"):
            grads = torch.autograd.grad(
                getattr(terms, key), params, retain_graph=True, allow_unused=True
            )
            norms[key] = math.sqrt(
                sum(
                    float(g.detach().float().square().sum())
                    for g in grads
                    if g is not None
                )
            )
            assert math.isfinite(norms[key]) and norms[key] > 0
        loss.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, raw["grad_clip"])
        rt.optimizer.step()
        records.append(
            dict(
                step=i,
                batch=b.batch_size,
                loss=float(loss.loss.detach()),
                gradient_norms=norms,
            )
        )
        m.r50_last_terms = None
    assert len(records) == 4 and all(x["batch"] == 64 for x in records)
    checkpoint = Path(raw["output_dir"]) / "smoke_only.pt"
    torch.save(dict(model=m.state_dict(), config=raw, epoch=0), checkpoint)
    metrics = evaluate_checkpoint(
        checkpoint, raw, Path(raw["output_dir"]) / "official_validation"
    )
    receipt = dict(
        status="passed",
        steps=records,
        metrics=metrics["metrics"],
        query_count=metrics["diagnostics"]["query_count"],
        scope="Engineering smoke only, not a formal result; never resume training from this checkpoint",
    )
    assert receipt["query_count"] == 1550
    (Path(raw["output_dir"]) / "smoke_receipt.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print(json.dumps(receipt))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=["train", "smoke", "evaluate", "delete-fields", "verify"]
    )
    p.add_argument("--seed", type=int, default=2041)
    p.add_argument("--output", type=Path, default=ROOT / "runs/seed2041")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--split", choices=["val", "test"], default="val")
    a = p.parse_args()
    if a.command == "verify":
        manifest = json.loads((ROOT / "FILE_SHA256.json").read_text())
        for name, expected in manifest.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, (
                name
            )
        print(f"Verified {len(manifest)} distributed files")
        return
    check_data()
    raw = materialize(a.seed, a.output)
    if a.command == "train":
        cfg = save_config(raw, create_output=False)
        from eventfieldnet.train import main as train_main

        sys.argv = [sys.argv[0], "run", "--config", str(cfg)]
        train_main()
    elif a.command == "smoke":
        smoke(raw, save_config(raw))
    else:
        if not a.checkpoint:
            p.error("--checkpoint is required")
        a.output.mkdir(parents=True, exist_ok=False)
        for tag in (
            ["full", "E", "S", "T"] if a.command == "delete-fields" else ["full"]
        ):
            result = evaluate_checkpoint(
                a.checkpoint,
                raw,
                a.output / tag,
                None if tag == "full" else tag,
                split=a.split,
            )
            print(tag, result["metrics"]["MR-mAP-Full_Avg"])


if __name__ == "__main__":
    main()
