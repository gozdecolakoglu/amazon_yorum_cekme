from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from amazon_iphone15_reviews import (
    extract_reviews,
    extract_reviews_from_csv,
    load_stop_words,
    preprocess_review,
    write_reviews_csv,
)


# Sahte bağımlılıklar, testlerin Zemberek/NLTK dış verisine ihtiyaç duymadan çalışmasını sağlar.
class FakeStemmer:
    def stem(self, word: str) -> str:
        return f"stem-{word}"


class FakeMorphology:
    def analyze(self, word: str) -> SimpleNamespace:
        # Yalnızca "güzel" biliniyormuş gibi davranarak bilinen/bilinmeyen sözcük yollarını kapsar.
        if word == "güzel":
            result = SimpleNamespace(
                is_unknown=lambda: False,
                item=SimpleNamespace(lemma="güzel"),
            )
            return SimpleNamespace(analysis_results=[result])
        return SimpleNamespace(analysis_results=[])


class AmazonIphone15ReviewsTests(unittest.TestCase):
    def test_extracts_only_reviews_from_iphone_15_html(self) -> None:
        # Bir klasördeki hedef ürün ile Pro sayfası birlikte bulunur; yalnızca hedef model dönmelidir.
        with TemporaryDirectory() as directory:
            source_dir = Path(directory)
            (source_dir / "reviews.html").write_text(
                """
                <html><head><title>Apple iPhone 15 128 GB yorumları</title></head>
                <body>
                  <div data-hook="review" id="R1">
                    <a data-hook="review-author">Ayşe</a>
                    <span data-hook="review-date">16 Ekim 2026</span>
                    <span data-hook="review-star-rating">5 yıldız</span>
                    <a data-hook="review-title">Harika</a>
                    <span data-hook="review-body">Telefon çok güzel!</span>
                  </div>
                </body></html>
                """,
                encoding="utf-8",
            )
            (source_dir / "iphone-15-pro.html").write_text(
                """
                <html><head><title>Apple iPhone 15 Pro yorumları</title></head>
                <body><div data-hook="review">
                  <span data-hook="review-body">Başka model.</span>
                </div></body></html>
                """,
                encoding="utf-8",
            )

            reviews = extract_reviews(source_dir)

        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["review_id"], "R1")
        self.assertEqual(reviews[0]["reviewer"], "Ayşe")
        self.assertEqual(reviews[0]["review"], "Telefon çok güzel!")

    def test_preprocesses_all_requested_stages(self) -> None:
        # Her ara çıktı kontrol edilerek ön işleme sırasındaki gerilemeler yakalanır.
        result = preprocess_review(
            "İYİ, bir telefon güzel! X",
            FakeMorphology(),
            load_stop_words() | {"iyi"},
            FakeStemmer(),
        )

        self.assertEqual(result["lowercase"], "iyi, bir telefon güzel! x")
        self.assertEqual(result["punctuation_cleaned"], "iyi bir telefon güzel x")
        self.assertEqual(result["tokens"], "iyi bir telefon güzel x")
        self.assertEqual(result["stopword_removed_tokens"], "telefon güzel x")
        self.assertEqual(result["stemmed_tokens"], "stem-telefon stem-güzel stem-x")
        self.assertEqual(result["single_character_removed_tokens"], "telefon güzel")
        self.assertEqual(result["lemmatized_tokens"], "telefon güzel")

    def test_reports_when_no_iphone_reviews_are_available(self) -> None:
        # Boş kaynakta açık hata verilmesi, sessizce boş çıktı üretilmesini önler.
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No iPhone 15 reviews"):
                extract_reviews(Path(directory))

    def test_extracts_iphone_15_rows_from_csv(self) -> None:
        # Türkçe başlık eşlemesi ve Pro modeli filtresi aynı küçük CSV örneğinde doğrulanır.
        with TemporaryDirectory() as directory:
            source_file = Path(directory) / "amazon_yorumları_cekilenler.csv"
            source_file.write_text(
                "urun_adi,yorum,puan,baslik\n"
                "Apple iPhone 15,Çok güzel telefon!,5,Harika\n"
                "Apple iPhone 15 Pro,Pro modeli,4,Pro\n",
                encoding="utf-8-sig",
            )

            reviews = extract_reviews_from_csv(source_file)

        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["review"], "Çok güzel telefon!")
        self.assertEqual(reviews[0]["rating"], "5")
        self.assertEqual(reviews[0]["title"], "Harika")

    def test_writes_extracted_and_preprocessed_fields_to_csv(self) -> None:
        # Yazılan CSV tekrar okunarak hem kaynak alanların hem işlenmiş metnin saklandığı doğrulanır.
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "reviews.csv"
            write_reviews_csv(
                [{"review_id": "R1", "review": "Telefon güzel"}],
                output_path,
                FakeMorphology(),
                load_stop_words(),
            )

            with output_path.open(encoding="utf-8-sig", newline="") as output_file:
                row = next(csv.DictReader(output_file))

        self.assertEqual(row["review_id"], "R1")
        self.assertEqual(row["tokens"], "telefon güzel")
        self.assertEqual(row["lemmatized_tokens"], "telefon güzel")


if __name__ == "__main__":
    unittest.main()
