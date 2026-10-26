from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from amazon_zero_shot_sentiment import (
    ASPECTS,
    DEFAULT_MODEL,
    analysis_fields,
    candidate_labels,
    classify_reviews,
    write_analysis,
)


class FakeZeroShotClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        texts: list[str],
        *,
        candidate_labels: list[str],
        hypothesis_template: str,
        multi_label: bool,
        batch_size: int,
    ) -> list[dict[str, object]]:
        self.calls += 1
        self.asserted_options = (hypothesis_template, multi_label, batch_size)
        outputs = []
        for text in texts:
            scores = {label: 0.1 for label in candidate_labels}
            if text == "good long review":
                scores[candidate_labels[0]] = 0.8
                scores[candidate_labels[1]] = 0.7
                scores[candidate_labels[2]] = 0.9
            if text == "bad long review":
                scores[candidate_labels[5]] = 0.95
            outputs.append(
                {
                    "labels": sorted(candidate_labels, key=scores.get, reverse=True),
                    "scores": sorted(scores.values(), reverse=True),
                }
            )
        return outputs


class ZeroShotSentimentTests(unittest.TestCase):
    def test_builds_paired_candidate_labels_for_each_aspect(self) -> None:
        labels = candidate_labels()

        self.assertEqual(len(ASPECTS), 5)
        self.assertEqual(len(labels), 10)
        self.assertIn("telefonun performansı ve pil ömrü açısından olumlu bir deneyim", labels)
        self.assertIn("telefonun performansı ve pil ömrü açısından olumsuz bir deneyim", labels)

    def test_classifies_multiple_aspects_and_excludes_short_reviews(self) -> None:
        classifier = FakeZeroShotClassifier()
        results = classify_reviews(
            [
                {"review_id": "R1", "review": "good long review"},
                {"review_id": "R2", "review": "bad long review"},
                {"review_id": "R3", "review": "Güzel"},
            ],
            classifier,
            batch_size=2,
        )

        self.assertEqual(classifier.calls, 1)
        self.assertEqual(results[0]["device_quality_sentiment"], "mixed")
        self.assertEqual(results[0]["device_quality_positive_score"], 0.8)
        self.assertEqual(results[0]["device_quality_negative_score"], 0.7)
        self.assertEqual(results[0]["performance_battery_sentiment"], "positive")
        self.assertEqual(results[1]["screen_camera_sentiment"], "negative")
        self.assertEqual(results[2]["analysis_status"], "short_review")
        self.assertEqual(results[2]["device_quality_positive_score"], "")
        self.assertEqual(results[2]["device_quality_sentiment"], "short_review")
        self.assertEqual(results[0]["model"], DEFAULT_MODEL)
        self.assertEqual(results[0]["threshold"], 0.5)
        self.assertEqual(classifier.asserted_options[1], True)

    def test_rejects_invalid_threshold_or_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "Eşik"):
            classify_reviews([{"review": "a longer comment"}], object(), threshold=1)
        with self.assertRaisesRegex(ValueError, "yorum metni boş"):
            classify_reviews([{"review": " "}], object())

    def test_writes_analysis_csv_with_all_aspect_scores(self) -> None:
        classifier = FakeZeroShotClassifier()
        rows = classify_reviews([{"review_id": "R1", "review": "good long review"}], classifier)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "analysis.csv"
            write_analysis(rows, output)
            with output.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                saved_row = next(reader)

        self.assertEqual(tuple(reader.fieldnames or ()), analysis_fields())
        self.assertEqual(saved_row["device_quality_sentiment"], "mixed")
        self.assertEqual(saved_row["screen_camera_positive_score"], "0.1")


if __name__ == "__main__":
    unittest.main()
