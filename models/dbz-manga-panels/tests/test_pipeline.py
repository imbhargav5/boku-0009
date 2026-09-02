from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


MODEL_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_module("dbz_prepare", MODEL_ROOT / "prepare_dataset.py")
evaluate = load_module("dbz_evaluate", MODEL_ROOT / "evaluate.py")


class PrepareDatasetTests(unittest.TestCase):
    def test_long_page_segments_cover_the_source_once(self):
        boxes = prepare.segment_boxes(600, 2400, 4)
        self.assertEqual(boxes, [(0, 0, 600, 600), (0, 600, 600, 1200), (0, 1200, 600, 1800), (0, 1800, 600, 2400)])

    def test_square_grayscale_has_expected_shape_and_information(self):
        image = Image.new("RGB", (300, 900), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 280, 880), outline="black", width=12)
        prepared = prepare.square_grayscale(image, 512)
        self.assertEqual(prepared.size, (512, 512))
        self.assertEqual(prepared.mode, "RGB")
        self.assertGreater(prepare.visual_stats(prepared)["ink_ratio"], 0.025)


class EvaluationTests(unittest.TestCase):
    def test_monochrome_feature_separates_grayscale_from_color(self):
        grayscale = Image.new("RGB", (64, 64), (128, 128, 128))
        color = Image.new("RGB", (64, 64), (255, 0, 0))
        self.assertGreater(evaluate.image_features(grayscale)["monochrome_score"], evaluate.image_features(color)["monochrome_score"])

    def test_style_distance_is_zero_for_identical_profiles(self):
        features = {
            "monochrome_score": 0.9,
            "black_ink_ratio": 0.2,
            "white_paper_ratio": 0.5,
            "contrast": 0.7,
            "edge_density": 0.3,
        }
        self.assertEqual(evaluate.style_distance(features, features), 0.0)


if __name__ == "__main__":
    unittest.main()
