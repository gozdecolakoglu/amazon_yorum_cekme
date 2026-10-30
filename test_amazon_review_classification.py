from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from amazon_review_classification import (
    _evaluation_metrics,
    load_classification_data,
    run_classical_models,
    run_stemming_comparison,
    run_training_size_comparison,
    sentiment_label,
    split_data,
    write_results,
    write_stemming_comparison,
    write_training_size_comparison,
)


# Testler küçük yapay örneklerle veri indirmeden sınıflandırma kurallarını doğrular.
class AmazonReviewClassificationTests(unittest.TestCase):
    def test_calculates_macro_metrics_and_one_vs_rest_roc_auc(self) -> None:
        metrics = _evaluation_metrics(
            ["olumsuz", "olumsuz", "olumlu", "olumlu"],
            ["olumsuz", "olumlu", "olumlu", "olumsuz"],
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
            ["olumsuz", "olumlu"],
        )

        self.assertEqual(metrics["test_accuracy"], 0.5)
        self.assertEqual(metrics["test_precision"], 0.5)
        self.assertEqual(metrics["test_recall"], 0.5)
        self.assertEqual(metrics["test_f1"], 0.5)
        self.assertEqual(metrics["test_roc_auc"], 1.0)

    def test_classifies_sentiment_from_review_words(self) -> None:
        # Duygu; yıldızdan değil, Türkçe karakterleri sadeleştirilmiş köklerden ve olumsuzluktan çıkarılır.
        self.assertEqual(sentiment_label("Ürün çok güzel, beğendim ve sağlam."), "olumlu")
        self.assertEqual(sentiment_label("Ürün berbat, bozuk ve çok kötü."), "olumsuz")
        self.assertEqual(sentiment_label("Telefon güzel değil."), "olumsuz")
        self.assertEqual(sentiment_label("Kargo sorunu yok, ürün sorunsuz."), "olumlu")

    def test_unmatched_or_conflicting_words_are_neutral(self) -> None:
        # Duygu işareti bulunmaması veya eşit olumlu/olumsuz işaret nötr sınıfına karşılık gelir.
        self.assertEqual(sentiment_label("Telefon bugün geldi."), "nötr")
        self.assertEqual(sentiment_label("Ürün güzel ama pahalı."), "nötr")

    def test_loads_text_and_derived_labels_without_rating_column(self) -> None:
        # Yalnızca yorum sütunu kullanılması, yıldız puanına bağımlılığın kaldırıldığını doğrular.
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=("review",))
                writer.writeheader()
                writer.writerow({"review": "Ürün kötü ve bozuk"})
                writer.writerow({"review": "Ürün çok iyi"})

            texts, labels = load_classification_data(path)

        self.assertEqual(texts, ["Ürün kötü ve bozuk", "Ürün çok iyi"])
        self.assertEqual(labels, ["olumsuz", "olumlu"])

    def test_classical_models_report_accuracy_and_test_metrics(self) -> None:
        # Her sınıftan yeterli örnek, stratified bölme ve dört modelin metrik üretmesini sağlar.
        texts = [
            f"{sentiment} ürün {index}"
            for sentiment in ("olumsuz", "nötr", "olumlu")
            for index in range(12)
        ]
        labels = [
            sentiment
            for sentiment in ("olumsuz", "nötr", "olumlu")
            for _ in range(12)
        ]
        x_train, x_test, y_train, y_test = split_data(texts, labels)

        results = run_classical_models(x_train, y_train, x_test, y_test)

        # Doğruluk oranlarının olasılık aralığında olması, metriklerin geçerli olduğuna işaret eder.
        self.assertEqual(len(results), 4)
        for result in results:
            for metric in (
                "train_accuracy",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
            ):
                self.assertGreaterEqual(result[metric], 0.0)
                self.assertLessEqual(result[metric], 1.0)

    def test_classical_models_support_binary_roc_auc(self) -> None:
        texts = [
            f"{sentiment} ürün {index}"
            for sentiment in ("olumsuz", "olumlu")
            for index in range(12)
        ]
        labels = [
            sentiment
            for sentiment in ("olumsuz", "olumlu")
            for _ in range(12)
        ]
        x_train, x_test, y_train, y_test = split_data(texts, labels)

        results = run_classical_models(x_train, y_train, x_test, y_test)

        self.assertEqual(len(results), 4)
        for result in results:
            self.assertIn("test_roc_auc", result)

    def test_writes_all_evaluation_metrics_to_csv(self) -> None:
        expected_fields = [
            "model",
            "train_accuracy",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
        ]
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "metrics.csv"
            write_results(
                [
                    {
                        "model": "Model",
                        "train_accuracy": 1.0,
                        "test_accuracy": 0.8,
                        "test_precision": 0.7,
                        "test_recall": 0.6,
                        "test_f1": 0.65,
                        "test_roc_auc": 0.9,
                    }
                ],
                output_path,
            )

            with output_path.open(encoding="utf-8-sig", newline="") as input_file:
                reader = csv.DictReader(input_file)
                row = next(reader)

        self.assertEqual(reader.fieldnames, expected_fields)
        self.assertEqual(row["test_roc_auc"], "0.9")

    def test_compares_stemming_with_the_same_stratified_split(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "processed.csv"
            with input_path.open("w", encoding="utf-8-sig", newline="") as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=("review", "stopword_removed_tokens", "stemmed_tokens"),
                )
                writer.writeheader()
                for index in range(15):
                    writer.writerow(
                        {
                            "review": f"Ürün güzel ve hızlı memnun {index}",
                            "stopword_removed_tokens": f"guzel hizli memnun {index}",
                            "stemmed_tokens": f"guzel hiz memnun {index}",
                        }
                    )
                    writer.writerow(
                        {
                            "review": f"Ürün berbat ve bozuk kötü {index}",
                            "stopword_removed_tokens": f"berbat bozuk kotu {index}",
                            "stemmed_tokens": f"berbat bozuk kot {index}",
                        }
                    )
                    writer.writerow(
                        {
                            "review": f"Telefon bugün geldi şimdi {index}",
                            "stopword_removed_tokens": f"telefon bugun geldi simdi {index}",
                            "stemmed_tokens": f"telefon bugun gel simdi {index}",
                        }
                    )

            results = run_stemming_comparison(input_path)

            self.assertEqual(len(results), 8)
            for model in (
                "Naive Bayes",
                "Lojistik Regresyon",
                "Support Vector Machine",
                "Random Forest",
            ):
                pair = [row for row in results if row["model"] == model]
                self.assertEqual(
                    {row["preprocessing"] for row in pair},
                    {"Stemming yok", "Stemming var"},
                )
            for row in results:
                self.assertGreaterEqual(row["test_f1"], 0.0)
                self.assertLessEqual(row["test_f1"], 1.0)

            output_path = Path(directory) / "comparison.csv"
            write_stemming_comparison(results, output_path)
            with output_path.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                saved_rows = list(reader)

        self.assertEqual(len(saved_rows), 8)
        self.assertIn("preprocessing", reader.fieldnames or ())
        self.assertIn("test_f1", reader.fieldnames or ())

    def test_compares_classical_models_across_training_sizes(self) -> None:
        texts = [
            f"{sentiment} ürün yorum {index}"
            for sentiment in ("olumsuz", "nötr", "olumlu")
            for index in range(20)
        ]
        labels = [
            sentiment
            for sentiment in ("olumsuz", "nötr", "olumlu")
            for _ in range(20)
        ]

        results = run_training_size_comparison(texts, labels)

        self.assertEqual(len(results), 24)
        for training_size in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
            split_results = [
                row for row in results if row["training_size"] == training_size
            ]
            expected_split = split_data(
                texts,
                labels,
                test_size=1 - training_size,
            )
            self.assertEqual(len(split_results), 4)
            self.assertEqual(
                {row["train_count"] for row in split_results},
                {len(expected_split[0])},
            )
            self.assertEqual(
                {row["test_count"] for row in split_results},
                {len(expected_split[1])},
            )

    def test_writes_training_size_metrics_to_csv(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "training-size.csv"
            write_training_size_comparison(
                [
                    {
                        "training_size": 0.6,
                        "train_count": 60,
                        "test_count": 40,
                        "model": "Model",
                        "train_accuracy": 1.0,
                        "test_accuracy": 0.8,
                        "test_precision": 0.7,
                        "test_recall": 0.6,
                        "test_f1": 0.65,
                        "test_roc_auc": 0.9,
                    }
                ],
                output_path,
            )

            with output_path.open(encoding="utf-8-sig", newline="") as input_file:
                reader = csv.DictReader(input_file)
                row = next(reader)

        self.assertEqual(reader.fieldnames[0:4], ["training_size", "train_count", "test_count", "model"])
        self.assertEqual(row["training_size"], "0.6")
        self.assertEqual(row["test_count"], "40")

if __name__ == "__main__":
    unittest.main()
