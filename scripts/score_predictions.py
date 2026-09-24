"""Score saved NMS predictions with the pinned Moment-DETR evaluator."""

from pathlib import Path
import argparse, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external"))
from standalone_eval.eval import eval_moment_retrieval

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    predictions = json.loads(a.predictions.read_text())["nms"]
    targets = [
        json.loads(x) for x in a.annotations.read_text().splitlines() if x.strip()
    ]
    assert len(predictions) == len(targets) == len({x["qid"] for x in predictions})
    assert {(x["qid"], x["vid"]) for x in predictions} == {
        (x["qid"], x["vid"]) for x in targets
    }
    result = eval_moment_retrieval(predictions, targets)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
