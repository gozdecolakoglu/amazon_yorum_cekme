"""Run paper-inspired, multi-label zero-shot aspect sentiment on iPhone reviews."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence


DEFAULT_INPUT = Path("iphone15_reviews_processed.csv")
DEFAULT_OUTPUT = Path("iphone15_sentiment_analysis.csv")
DEFAULT_MODEL = "vicgalle/xlm-roberta-large-xnli-anli"
DEFAULT_THRESHOLD = 0.5
MINIMUM_WORD_COUNT = 3
HYPOTHESIS_TEMPLATE = "Bu telefon yorumu {} hakkında."
ASPECTS = (
    ("device_quality", "Cihaz kalitesi ve tasarım", "cihaz kalitesi ve tasarımı"),
    ("performance_battery", "Performans ve pil", "telefonun performansı ve pil ömrü"),
    ("screen_camera", "Ekran ve kamera", "telefonun ekranı ve kamerası"),
    ("seller_delivery", "Satıcı ve teslimat", "satıcı ve kargo deneyimi"),
    ("price_value", "Fiyat ve değer", "telefonun fiyatı ve parasının karşılığı"),
)
POLARITIES = ("positive", "negative")


def analysis_fields() -> tuple[str, ...]:
    fields = [
        "review_number",
        "review_id",
        "analysis_status",
        "model",
        "word_count",
        "threshold",
    ]
    for aspect_id, _, _ in ASPECTS:
        fields.append(f"{aspect_id}_sentiment")
        fields.extend(f"{aspect_id}_{polarity}_score" for polarity in POLARITIES)
    return tuple(fields)


def load_reviews(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Yorum CSV dosyası bulunamadı: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"review", "review_id"}.issubset(reader.fieldnames):
            raise ValueError("Yorum CSV'sinde review ve review_id sütunları olmalı.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Yorum CSV dosyasında yorum bulunamadı: {path}")
    for row_number, row in enumerate(rows, start=2):
        if not (row.get("review") or "").strip():
            raise ValueError(f"{row_number}. satırdaki yorum metni boş.")
    return rows


def candidate_labels() -> tuple[str, ...]:
    return tuple(
        f"{topic} açısından {polarity} bir deneyim"
        for _, _, topic in ASPECTS
        for polarity in ("olumlu", "olumsuz")
    )


def classify_reviews(
    reviews: Sequence[dict[str, str]],
    classifier: Any,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 4,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, str | int | float]]:
    """Classify each aspect's positive and negative sentiment independently."""
    if not 0 < threshold < 1:
        raise ValueError("Eşik değeri 0 ile 1 arasında olmalıdır.")
    if batch_size < 1:
        raise ValueError("Batch size en az 1 olmalıdır.")

    eligible: list[tuple[int, str]] = []
    results: list[dict[str, str | int | float]] = []
    for review_number, review in enumerate(reviews, start=1):
        text = (review.get("review") or "").strip()
        if not text:
            raise ValueError(f"{review_number}. satırdaki yorum metni boş.")
        word_count = len(text.split())
        result: dict[str, str | int | float] = {
            "review_number": review_number,
            "review_id": review.get("review_id") or "",
            "analysis_status": "eligible" if word_count >= MINIMUM_WORD_COUNT else "short_review",
            "model": model_name,
            "word_count": word_count,
            "threshold": threshold,
        }
        for aspect_id, _, _ in ASPECTS:
            result[f"{aspect_id}_sentiment"] = (
                "not_mentioned" if word_count >= MINIMUM_WORD_COUNT else "short_review"
            )
            for polarity in POLARITIES:
                result[f"{aspect_id}_{polarity}_score"] = ""
        results.append(result)
        if word_count >= MINIMUM_WORD_COUNT:
            eligible.append((review_number - 1, text))

    labels = candidate_labels()
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start : start + batch_size]
        predictions = classifier(
            [text for _, text in batch],
            candidate_labels=list(labels),
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,
            batch_size=batch_size,
        )
        if isinstance(predictions, dict):
            predictions = [predictions]
        if len(predictions) != len(batch):
            raise ValueError("Sıfır-atış modeli beklenmeyen sayıda sonuç döndürdü.")
        for (result_index, _), prediction in zip(batch, predictions, strict=True):
            score_map = dict(zip(prediction["labels"], prediction["scores"], strict=True))
            if set(score_map) != set(labels):
                raise ValueError("Sıfır-atış modeli tüm hedef sınıflar için puan üretmedi.")
            result = results[result_index]
            for aspect_id, _, topic in ASPECTS:
                for polarity, turkish_polarity in (
                    ("positive", "olumlu"),
                    ("negative", "olumsuz"),
                ):
                    label = f"{topic} açısından {turkish_polarity} bir deneyim"
                    score = float(score_map[label])
                    if not 0 <= score <= 1:
                        raise ValueError(f"Model geçersiz bir sınıf puanı üretti: {score}.")
                    result[f"{aspect_id}_{polarity}_score"] = score
                positive = float(result[f"{aspect_id}_positive_score"])
                negative = float(result[f"{aspect_id}_negative_score"])
                if positive >= threshold and negative >= threshold:
                    aspect_sentiment = "mixed"
                elif positive >= threshold:
                    aspect_sentiment = "positive"
                elif negative >= threshold:
                    aspect_sentiment = "negative"
                else:
                    aspect_sentiment = "not_mentioned"
                result[f"{aspect_id}_sentiment"] = aspect_sentiment
    return results


def write_analysis(rows: Sequence[dict[str, str | int | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=analysis_fields())
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    input_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 4,
) -> list[dict[str, str | int | float]]:
    reviews = load_reviews(input_path)
    import torch
    from transformers import pipeline

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )
    results = classify_reviews(
        reviews,
        classifier,
        threshold,
        batch_size,
        model_name,
    )
    write_analysis(results, output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="iPhone 15 yorumlarını çok etiketli sıfır-atış hedef tabanlı duygu analiziyle inceler."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="İşlenmiş yorum CSV dosyası.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Duygu analizi sonuç CSV dosyası.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face NLI modeli.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Hedef etiketini olumlu/olumsuz sayma eşiği (varsayılan: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="CPU belleğini gözeten metin batch boyutu (varsayılan: 4).",
    )
    args = parser.parse_args()
    results = run_analysis(
        args.input,
        args.output,
        model_name=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    eligible_count = sum(row["analysis_status"] == "eligible" for row in results)
    print(f"Analiz tamamlandı: {eligible_count}/{len(results)} yorum")
    print(f"Sonuç CSV'si: {args.output}")
    print(f"Model: {args.model} | Eşik: {args.threshold:.2f}")


if __name__ == "__main__":
    main()
