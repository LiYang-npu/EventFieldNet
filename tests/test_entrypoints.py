"""Regression checks for checkpoint metadata migration and output isolation."""

import copy, json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import normalize_factory, save_config, materialize


class EntryPointTests(unittest.TestCase):
    def test_current_factory_is_not_translated_twice(self):
        factory = json.loads((ROOT / "configs/qvhighlights.json").read_text())[
            "model_factory"
        ]
        self.assertEqual(normalize_factory(factory), factory)
        self.assertEqual(normalize_factory(normalize_factory(factory)), factory)

    def test_reference_factory_maps_to_paper_factory(self):
        current = json.loads((ROOT / "configs/qvhighlights.json").read_text())[
            "model_factory"
        ]
        old = copy.deepcopy(current)
        old["target"] = "trifield_round32.r68:build_model"
        self.assertEqual(normalize_factory(old), current)

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = materialize(2041, Path(directory) / "run")
            save_config(raw)
            with self.assertRaises(FileExistsError):
                save_config(raw)

    def test_five_reported_seed_configs_start_fresh(self):
        for seed in range(2041, 2046):
            raw = materialize(seed, ROOT / "runs" / str(seed))
            self.assertEqual(raw["seed"], seed)
            self.assertEqual(raw["epochs"], 24)
            self.assertFalse(raw["resume"])
            self.assertIsNone(raw["base_checkpoint"])
            self.assertEqual(
                raw["model_factory"]["kwargs"]["followup_spec"],
                {"edge_route": True, "long_kl": False},
            )


if __name__ == "__main__":
    unittest.main()
