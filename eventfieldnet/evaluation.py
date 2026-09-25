"""Strict-FP32 moment retrieval with optional official highlight evaluation."""
import functools
import hashlib
import json
import sys
from pathlib import Path

from . import precision_evaluation as precision



def _task(raw):
    task = raw.get("task", "mr_hd")
    if task not in {"mr", "mr_hd"}:
        raise ValueError("task must be 'mr' or 'mr_hd'")
    enabled = raw.get("model_factory", {}).get("kwargs", {}).get("enable_highlight")
    if enabled is not None and (
        not isinstance(enabled, bool) or enabled != (task == "mr_hd")
    ):
        raise ValueError("task and model enable_highlight disagree")
    return task


def _mr_metrics(**kwargs):
    """Use exactly the MR members and defaults of the SG metric collection."""
    from torchmetrics import MetricCollection
    from sg_components.metrics.moments.metrics import MRAveragePrecision, MRRecallAt1

    return MetricCollection({
        **{f"MR-mAP-{name.title()}": MRAveragePrecision(window_range=name, **kwargs)
           for name in ("short", "middle", "long", "full")},
        **{f"MR-R1-{name.title()}": MRRecallAt1(window_range=name, **kwargs)
           for name in ("short", "middle", "long", "full")},
    })


def _config(payload, config_path):
    """Explicit portable config replaces historical machine paths, never weights."""
    if config_path is None:
        raw = dict(payload['config'])
    else:
        raw = json.loads(Path(config_path).read_text(encoding='utf-8'))
        if int(raw['seed']) != int(payload['config']['seed']):
            raise ValueError('Checkpoint and evaluation configuration seeds differ')
    target = raw['model_factory']['target']
    if target != 'eventfieldnet.evidence_supervision.model:build_model':
        raise ValueError('Use the portable EventFieldNet configuration for this checkpoint')
    original = payload.get('config', {}).get('model_factory', {})
    if original.get('target') not in {'ecf_study.model:build_model', 'eventfieldnet.evidence_supervision.model:build_model'} or original.get('kwargs', {}).get('cf_arm') != 'C02':
        raise ValueError('Expected a C02 checkpoint, not another recipe')
    if raw['model_factory']['kwargs'].get('cf_arm') != 'C02':
        raise ValueError('Expected the fixed C02 evaluation configuration')
    if _task(raw) != _task(payload.get("config", {})):
        raise ValueError("Checkpoint and evaluation tasks differ")
    return raw


def evaluate_checkpoint(*args, **kwargs):
    base = precision.original
    names = ("_evidence_scores", "_checkpoint_config", "_repository_metric_api",
             "_prediction", "_prediction_json")
    previous = {name: getattr(base, name) for name in names}
    active = {"task": None}

    def config(payload, config_path):
        raw = _config(payload, config_path)
        active["task"] = _task(raw)
        return raw

    def scores(output, row, clips, torch):
        if active["task"] == "mr":
            return None
        value = output.hd_logits[row, :clips].float()
        if value.numel() != clips or not torch.isfinite(value).all():
            raise ValueError('Invalid highlight head output')
        return value.detach().cpu()

    def metric_api(project_root):
        get_metrics, to_python, script = previous["_repository_metric_api"](project_root)
        if active["task"] == "mr":
            get_metrics = _mr_metrics
        return get_metrics, to_python, script

    def prediction(*args, **kwargs):
        value = previous["_prediction"](*args, **kwargs)
        if active["task"] == "mr":
            value.pop("pred_saliency_scores", None)
        return value

    def prediction_json(value):
        result = previous["_prediction_json"](value)
        if active["task"] == "mr":
            result.pop("pred_saliency_scores", None)
        return result

    base._evidence_scores = scores
    base._checkpoint_config = config
    base._repository_metric_api = metric_api
    base._prediction = prediction
    base._prediction_json = prediction_json
    try:
        result = precision.evaluate_checkpoint(*args, **kwargs)
    finally:
        for name, function in previous.items():
            setattr(base, name, function)

    if active["task"] == "mr":
        # The MR-only collection prevents HL/HD keys in every metric artifact.
        return result

    root, output = Path(kwargs['root']), Path(kwargs['output_dir'])
    annotation = root/'data/qvhighlights/annotation/highlight_val_release.jsonl'
    truth = [json.loads(line) for line in annotation.read_text().splitlines() if line.strip()]
    predictions = json.loads((output/'predictions.json').read_text())['nms']
    by_id = {str(row['qid']): row for row in truth}
    if len(predictions) != len(truth) or len({str(p['qid']) for p in predictions}) != len(truth):
        raise ValueError('Evaluation must cover every unique validation query')
    if {str(p['qid']) for p in predictions} != set(by_id):
        raise ValueError('Prediction and annotation query identities differ')
    for prediction in predictions:
        target = by_id[str(prediction['qid'])]
        if prediction['vid'] != target['vid']:
            raise ValueError('Video identity mismatch')
        clips = int(target['duration']/2)
        padded = len(prediction['pred_saliency_scores'])
        if not (clips <= padded <= clips+4 and padded % 4 == 0):
            raise ValueError('Incorrect highlight time axis')
        prediction['pred_saliency_scores'] = prediction['pred_saliency_scores'][:clips]
    sys.path.insert(0, str(root/'external'))
    from standalone_eval import eval as official
    evaluator_file = root/'external/standalone_eval/eval.py'
    if hashlib.sha256(evaluator_file.read_bytes()).hexdigest() != '679e43d8f83ca08a682dbe2e4e3f4629ddc2e6911981bcd0c0e54cf1c0cfabca':
        raise ValueError('Official evaluator source differs from the pinned version')
    original_ap = official.compute_hl_ap
    official.compute_hl_ap = functools.partial(original_ap, num_workers=1)
    try:
        highlight = official.eval_highlight(predictions, truth, verbose=False)
    finally:
        official.compute_hl_ap = original_ap
    for threshold, values in highlight.items():
        for name, value in values.items():
            result['metrics'][f'official_HD/{threshold}/{name}'] = value
    result['hd'] = {'metrics': highlight, 'readout': 'fusion_highlight_head',
                    'annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest()}
    (output/'hd_predictions.json').write_text(json.dumps(predictions))
    (output/'hd_receipt.json').write_text(json.dumps(result['hd'], indent=2))
    (output/'metrics.json').write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    old = precision.original.evaluate_checkpoint
    precision.original.evaluate_checkpoint = evaluate_checkpoint
    try:
        return precision.original.main(argv)
    finally:
        precision.original.evaluate_checkpoint = old


if __name__ == '__main__':
    raise SystemExit(main())
