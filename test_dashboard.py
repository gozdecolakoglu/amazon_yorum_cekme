from __future__ import annotations

import unittest

from amazon_review_classification import TRAINING_SIZE_PROPORTIONS
from amazon_zero_shot_sentiment import ASPECTS, DEFAULT_MODEL, analysis_fields
from dashboard import build_dashboard_data, parse_rating, parse_review_month


def make_training_size_rows() -> list[dict[str, str]]:
    rows = []
    metrics = (
        "train_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    )
    models = (
        "Naive Bayes",
        "Lojistik Regresyon",
        "Support Vector Machine",
        "Random Forest",
    )
    for training_size in TRAINING_SIZE_PROPORTIONS:
        for index, model in enumerate(models):
            row = {
                "training_size": str(training_size),
                "train_count": str(round(training_size * 100)),
                "test_count": str(100 - round(training_size * 100)),
                "model": model,
            }
            row.update({metric: str(0.5 + index * 0.05) for metric in metrics})
            rows.append(row)
    return rows


def make_stemming_rows() -> list[dict[str, str]]:
    fields = (
        "train_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    )
    rows = []
    for preprocessing, offset in (("Stemming yok", 0), ("Stemming var", 0.1)):
        row = {"model": "Model", "preprocessing": preprocessing}
        row.update(
            {
                metric: str(0.5 + offset)
                for metric in fields
            }
        )
        rows.append(row)
    return rows


def make_bow_tfidf_rows() -> list[dict[str, str]]:
    fields = (
        "train_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    )
    rows = []
    for vectorizer, offset in (("Bag of Words (BoW)", 0), ("TF-IDF", 0.08)):
        row = {"model": "Model", "vectorizer": vectorizer}
        row.update(
            {
                metric: str(0.6 + offset)
                for metric in fields
            }
        )
        rows.append(row)
    return rows


def make_bow_tfidf_term_rows() -> list[dict[str, str]]:
    return [
        {
            "term": "amazon",
            "ngram_size": "1",
            "term_frequency": "37",
            "document_frequency": "27",
            "document_share": "0.27",
            "mean_tfidf": "0.027",
            "max_tfidf": "0.42",
            "idf": "2.28",
            "vocab_size": "3548",
            "unigram_count": "1333",
            "bigram_count": "2215",
            "non_zero_count": "4658",
            "sparsity": "0.9869",
        },
        {
            "term": "elime ulaştı",
            "ngram_size": "2",
            "term_frequency": "7",
            "document_frequency": "6",
            "document_share": "0.06",
            "mean_tfidf": "0.012",
            "max_tfidf": "0.33",
            "idf": "3.67",
            "vocab_size": "3548",
            "unigram_count": "1333",
            "bigram_count": "2215",
            "non_zero_count": "4658",
            "sparsity": "0.9869",
        },
    ]


def make_aspect_row(review_number: int, short_review: bool = False) -> dict[str, str]:
    row = {
        "review_number": str(review_number),
        "review_id": f"R{review_number}",
        "analysis_status": "short_review" if short_review else "eligible",
        "model": DEFAULT_MODEL,
        "word_count": "1" if short_review else "5",
        "threshold": "0.5",
    }
    for aspect_id, _, _ in ASPECTS:
        row[f"{aspect_id}_sentiment"] = "short_review" if short_review else "not_mentioned"
        row[f"{aspect_id}_positive_score"] = "" if short_review else "0.05"
        row[f"{aspect_id}_negative_score"] = "" if short_review else "0.1"
    if not short_review:
        row["device_quality_sentiment"] = "positive"
        row["device_quality_positive_score"] = "0.9"
    return row


class DashboardTests(unittest.TestCase):
    def test_parses_localized_rating(self) -> None:
        self.assertEqual(parse_rating("5 yıldız üzerinden 4,0"), 4.0)
        self.assertIsNone(parse_rating("puan yok"))

    def test_parses_turkish_month_and_numeric_dates(self) -> None:
        self.assertEqual(
            parse_review_month("Türkiye'de 15 Şubat 2024 tarihinde değerlendirildi"),
            "2024-02",
        )
        self.assertEqual(parse_review_month("15.02.2024"), "2024-02")
        self.assertIsNone(parse_review_month("tarih yok"))

    def test_aggregates_review_and_model_data(self) -> None:
        data = build_dashboard_data(
            [
                {
                    "reviewer": "Ada",
                    "date": "15 Şubat 2024",
                    "rating": "5 yıldız üzerinden 5,0",
                    "title": "Harika",
                    "review": "Ürün çok güzel ve sağlam.",
                },
                {
                    "reviewer": "",
                    "date": "16 Şubat 2024",
                    "rating": "5 yıldız üzerinden 1,0",
                    "title": "",
                    "review": "Ürün bozuk ve berbat.",
                },
            ],
            [("Model", {"train_accuracy": 0.9, "test_accuracy": 0.8})],
        )

        self.assertEqual(data["summary"]["review_count"], 2)
        self.assertEqual(data["summary"]["average_rating"], 3.0)
        self.assertEqual(data["ratings"][0], {"rating": 5, "count": 1})
        self.assertEqual(data["sentiments"][0], {"sentiment": "olumlu", "count": 1})
        self.assertEqual(data["sentiments"][2], {"sentiment": "olumsuz", "count": 1})
        self.assertEqual(data["months"], [{"month": "2024-02", "count": 2}])
        self.assertEqual(data["reviews"][1]["reviewer"], "Anonim")
        self.assertEqual(data["models"][0]["test_accuracy"], 0.8)

    def test_rejects_empty_review_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "boş bir review"):
            build_dashboard_data(
                [{"review": " ", "rating": "", "date": ""}],
                [],
            )

    def test_includes_multi_label_aspects_and_omits_short_comments(self) -> None:
        aspect_rows = [make_aspect_row(1), make_aspect_row(2, short_review=True)]
        self.assertEqual(set(aspect_rows[0]), set(analysis_fields()))
        data = build_dashboard_data(
            [
                {"review_id": "R1", "review": "Ürün güzel ve kaliteli.", "rating": "5 yıldız üzerinden 5,0"},
                {"review_id": "R2", "review": "Güzel.", "rating": "5 yıldız üzerinden 5,0"},
            ],
            [],
            aspect_rows,
        )

        analysis = data["aspect_analysis"]
        self.assertEqual(analysis["model"], DEFAULT_MODEL)
        self.assertEqual(analysis["eligible_count"], 1)
        self.assertEqual(analysis["short_review_count"], 1)
        self.assertEqual(analysis["aspects"][0]["positive_count"], 1)
        self.assertEqual(data["reviews"][0]["aspects"][0]["sentiment"], "positive")
        self.assertEqual(data["reviews"][1]["analysis_status"], "short_review")
        self.assertEqual(data["reviews"][1]["aspects"][0]["sentiment"], "short_review")

    def test_rejects_aspect_rows_that_do_not_match_reviews(self) -> None:
        with self.assertRaisesRegex(ValueError, "satır sayısı"):
            build_dashboard_data(
                [{"review": "Ürün çok güzel.", "rating": "", "date": ""}],
                [],
                [],
            )

    def test_includes_paired_stemming_metrics_and_metric_deltas(self) -> None:
        data = build_dashboard_data(
            [{"review": "Ürün çok güzel.", "rating": "", "date": ""}],
            [],
            stemming_rows=make_stemming_rows(),
        )

        comparison = data["stemming_comparison"]
        self.assertEqual(comparison["row_count"], 2)
        self.assertEqual(comparison["models"][0]["without_stemming"]["test_f1"], 0.5)
        self.assertEqual(comparison["models"][0]["with_stemming"]["test_f1"], 0.6)
        self.assertAlmostEqual(comparison["models"][0]["delta_f1"], 0.1)

    def test_rejects_unpaired_stemming_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "ikisi de bulunmalı"):
            build_dashboard_data(
                [{"review": "Ürün çok güzel.", "rating": "", "date": ""}],
                [],
                stemming_rows=make_stemming_rows()[:1],
            )

    def test_includes_training_size_comparison_for_all_models(self) -> None:
        data = build_dashboard_data(
            [{"review": "Ürün çok güzel.", "rating": "", "date": ""} for _ in range(100)],
            [],
            training_size_rows=make_training_size_rows(),
        )

        comparison = data["training_size_comparison"]
        self.assertEqual(
            [split["training_size"] for split in comparison["splits"]],
            list(TRAINING_SIZE_PROPORTIONS),
        )
        self.assertEqual(comparison["splits"][0]["train_count"], 60)
        self.assertEqual(comparison["splits"][0]["test_count"], 40)
        self.assertEqual(len(comparison["splits"][0]["models"]), 4)

    def test_rejects_incomplete_training_size_comparison(self) -> None:
        with self.assertRaisesRegex(ValueError, "dört klasik model"):
            build_dashboard_data(
                [{"review": "Ürün çok güzel.", "rating": "", "date": ""} for _ in range(100)],
                [],
                training_size_rows=make_training_size_rows()[:-1],
            )

    def test_includes_bow_tfidf_comparison_and_top_terms(self) -> None:
        data = build_dashboard_data(
            [{"review": "Ürün çok güzel.", "rating": "", "date": ""} for _ in range(10)],
            [],
            bow_tfidf_rows=make_bow_tfidf_rows(),
            bow_tfidf_term_rows=make_bow_tfidf_term_rows(),
        )

        comparison = data["bow_tfidf_comparison"]
        self.assertEqual(comparison["row_count"], 2)
        self.assertEqual(comparison["train_count"], 8)
        self.assertEqual(comparison["test_count"], 2)
        self.assertAlmostEqual(comparison["models"][0]["delta_f1"], 0.08)
        self.assertEqual(comparison["corpus_stats"]["vocab_size"], 3548)
        self.assertEqual(len(comparison["terms"]), 2)
        self.assertEqual(comparison["top_bow_terms"][0]["term"], "amazon")
        self.assertEqual(comparison["top_bigram_terms"][0]["term"], "elime ulaştı")

    def test_rejects_unpaired_bow_tfidf_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "ikisi de bulunmalı"):
            build_dashboard_data(
                [{"review": "Ürün çok güzel.", "rating": "", "date": ""}],
                [],
                bow_tfidf_rows=make_bow_tfidf_rows()[:1],
            )


if __name__ == "__main__":
    unittest.main()
