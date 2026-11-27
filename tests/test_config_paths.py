import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from freqssl.config import load_config
from freqssl.paths import REPO_ROOT, expand

ROOT = Path(__file__).resolve().parents[1]
PLAIN = "/data/zhp/dateset_spine copy/spider/slice_png_order"


class ConfigPathTests(unittest.TestCase):
    def test_plain_values_are_unchanged(self):
        self.assertEqual(expand({"a": PLAIN, "b": [1, "x"], "c": 0.5}), {"a": PLAIN, "b": [1, "x"], "c": 0.5})

    def test_default_and_environment_override(self):
        text = "${SPIDER_PNG_ROOT:-" + PLAIN + "}/train"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPIDER_PNG_ROOT", None)
            self.assertEqual(expand(text), PLAIN + "/train")
            os.environ["SPIDER_PNG_ROOT"] = "/cluster/spider"
            self.assertEqual(expand(text), "/cluster/spider/train")

    def test_repo_root_and_missing_variable(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REPO_ROOT", None)
            os.environ.pop("FTP_TEST_UNSET", None)
            self.assertEqual(expand("${REPO_ROOT}/data"), f"{REPO_ROOT}/data")
            with self.assertRaises(KeyError):
                expand("${FTP_TEST_UNSET}")

    def test_shipped_v3_configs_keep_original_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in ("SPIDER_PNG_ROOT", "REPO_ROOT"):
                os.environ.pop(name, None)
            for path in sorted((ROOT / "configs/v3").rglob("*.yaml")):
                with self.subTest(config=path.name):
                    cfg = load_config(path)
                    self.assertEqual(cfg.root_path, PLAIN)
                    self.assertNotIn("${", str(cfg.to_dict()))

    def test_override_reaches_loaded_config(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SPIDER_PNG_ROOT": directory}):
            cfg = load_config(ROOT / "configs/v3/mean_teacher.yaml")
            self.assertEqual(cfg.root_path, directory)


if __name__ == "__main__":
    unittest.main()
