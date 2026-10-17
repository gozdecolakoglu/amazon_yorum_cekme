from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from amazon_review_classification import (
    load_classification_data,
    run_classical_models,
    sentiment_label,
    split_data,
)


# Testler küçük yapay örneklerle veri indirmeden sınıflandırma kurallarını doğrular.
class AmazonReviewClassificationTests(unittest.TestCase):
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

    def test_classical_models_report_train_and_test_accuracy(self) -> None:
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
            self.assertGreaterEqual(result["train_accuracy"], 0.0)
            self.assertLessEqual(result["train_accuracy"], 1.0)
            self.assertGreaterEqual(result["test_accuracy"], 0.0)
            self.assertLessEqual(result["test_accuracy"], 1.0)

if __name__ == "__main__":
    unittest.main()
