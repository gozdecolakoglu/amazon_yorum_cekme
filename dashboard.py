"""Serve a local dashboard for the processed iPhone 15 review data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from amazon_review_classification import (
    BOW_TFIDF_TERM_FIELDS,
    SMOTE_COMPARISON_FIELDS,
    SMOTE_CONDITIONS,
    TRAINING_SIZE_PROPORTIONS,
    VECTORIZER_NAMES,
    sentiment_label,
)
from amazon_zero_shot_sentiment import ASPECTS, analysis_fields
from plot_classification_accuracy import METRICS, read_scores


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_REVIEWS = PROJECT_DIR / "iphone15_reviews_processed.csv"
DEFAULT_METRICS = PROJECT_DIR / "classification_accuracy.csv"
DEFAULT_ASPECT_ANALYSIS = PROJECT_DIR / "iphone15_sentiment_analysis.csv"
DEFAULT_STEMMING_COMPARISON = PROJECT_DIR / "classification_stemming_comparison.csv"
DEFAULT_TRAINING_SIZE_COMPARISON = PROJECT_DIR / "classification_training_size_comparison.csv"
DEFAULT_BOW_TFIDF_COMPARISON = PROJECT_DIR / "classification_bow_tfidf_comparison.csv"
DEFAULT_BOW_TFIDF_TERMS = PROJECT_DIR / "bow_tfidf_top_terms.csv"
DEFAULT_SMOTE_COMPARISON = PROJECT_DIR / "classification_smote_comparison.csv"
STEMMING_PREPROCESSING = ("Stemming yok", "Stemming var")
CLASSIFICATION_MODEL_NAMES = (
    "Naive Bayes",
    "Lojistik Regresyon",
    "Support Vector Machine",
    "Random Forest",
)
CLASSIFICATION_METRICS = (
    "train_accuracy",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
)
MONTHS = {
    "ocak": 1,
    "şubat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "eylül": 9,
    "ekim": 10,
    "kasım": 11,
    "aralık": 12,
}
RATING_PATTERN = re.compile(r"(\d+(?:[,.]\d+)?)\s*$")
NUMERIC_DATE_PATTERN = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")


def parse_rating(value: str) -> float | None:
    """Extract the selected score from Amazon's localized rating string."""
    score_text = value.rsplit("üzerinden", maxsplit=1)[-1]
    match = RATING_PATTERN.search(score_text.strip())
    if match is None:
        return None
    score = float(match.group(1).replace(",", "."))
    return score if 1 <= score <= 5 else None


def parse_review_month(value: str) -> str | None:
    """Return an ISO year-month for Turkish month names or numeric dates."""
    numeric_match = NUMERIC_DATE_PATTERN.search(value)
    if numeric_match is not None:
        day, month, year = (int(part) for part in numeric_match.groups())
        try:
            return date(year, month, day).strftime("%Y-%m")
        except ValueError:
            return None

    parts = value.lower().split()
    for index, part in enumerate(parts):
        month = MONTHS.get(part.strip(".,"))
        if month is None:
            continue
        year_match = next(
            (
                match
                for candidate in parts[index + 1 :]
                if (match := re.fullmatch(r"\d{4}", candidate.strip(".,:")))
                is not None
            ),
            None,
        )
        if year_match is not None:
            year = int(year_match.group())
            if 1 <= year <= 9999:
                return f"{year:04d}-{month:02d}"
    return None


def build_dashboard_data(
    review_rows: list[dict[str, str]],
    score_rows: list[tuple[str, dict[str, float]]],
    aspect_rows: list[dict[str, str]] | None = None,
    stemming_rows: list[dict[str, str]] | None = None,
    training_size_rows: list[dict[str, str]] | None = None,
    bow_tfidf_rows: list[dict[str, str]] | None = None,
    bow_tfidf_term_rows: list[dict[str, str]] | None = None,
    smote_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Aggregate review and model CSV data into the dashboard's JSON payload."""
    ratings: Counter[int] = Counter()
    sentiments: Counter[str] = Counter()
    months: Counter[str] = Counter()
    parsed_ratings: list[float] = []
    reviews = []
    aspect_reviews: dict[int, dict[str, Any]] = {}
    aspect_counts = {aspect_id: Counter() for aspect_id, _, _ in ASPECTS}
    eligible_count = 0
    short_review_count = 0
    aspect_model = None
    aspect_threshold = None
    stemming_comparison = None
    training_size_comparison = None
    bow_tfidf_comparison = None
    smote_comparison = None

    if stemming_rows is not None:
        comparison_by_model: dict[str, dict[str, dict[str, float]]] = {}
        for row_number, comparison in enumerate(stemming_rows, start=2):
            model = (comparison.get("model") or "").strip()
            preprocessing = comparison.get("preprocessing")
            if not model or preprocessing not in STEMMING_PREPROCESSING:
                raise ValueError(
                    f"Stemming karşılaştırma CSV'sinde {row_number}. satırın model/ön işleme bilgisi geçersiz."
                )
            model_results = comparison_by_model.setdefault(model, {})
            if preprocessing in model_results:
                raise ValueError(f"{model} modeli için tekrarlı {preprocessing} sonucu var.")
            metrics: dict[str, float] = {}
            for metric in CLASSIFICATION_METRICS:
                try:
                    value = float(comparison[metric])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Stemming karşılaştırma CSV'sinde {row_number}. satırdaki {metric} değeri geçersiz."
                    ) from error
                if not 0 <= value <= 1:
                    raise ValueError(f"{metric} değeri 0 ile 1 arasında olmalıdır.")
                metrics[metric] = value
            model_results[preprocessing] = metrics

        if not comparison_by_model:
            raise ValueError("Stemming karşılaştırma CSV'sinde model sonucu bulunamadı.")
        for model, model_results in comparison_by_model.items():
            if set(model_results) != set(STEMMING_PREPROCESSING):
                raise ValueError(
                    f"{model} modelinde stemming var/yok sonuçlarının ikisi de bulunmalı."
                )
        stemming_comparison = {
            "models": [
                {
                    "model": model,
                    "without_stemming": model_results["Stemming yok"],
                    "with_stemming": model_results["Stemming var"],
                    "delta_accuracy": (
                        model_results["Stemming var"]["test_accuracy"]
                        - model_results["Stemming yok"]["test_accuracy"]
                    ),
                    "delta_f1": (
                        model_results["Stemming var"]["test_f1"]
                        - model_results["Stemming yok"]["test_f1"]
                    ),
                }
                for model, model_results in comparison_by_model.items()
            ],
            "row_count": len(stemming_rows),
            "test_count": round(len(review_rows) * 0.2),
            "label_source": "Yorum metninden türetilen sözlük etiketleri",
        }

    if bow_tfidf_rows is not None:
        bow_tfidf_by_model: dict[str, dict[str, dict[str, float]]] = {}
        for row_number, comparison in enumerate(bow_tfidf_rows, start=2):
            model = (comparison.get("model") or "").strip()
            vectorizer = comparison.get("vectorizer")
            if not model or vectorizer not in VECTORIZER_NAMES:
                raise ValueError(
                    f"BoW/TF-IDF karşılaştırma CSV'sinde {row_number}. satırın model/vektörleştirici bilgisi geçersiz."
                )
            model_results = bow_tfidf_by_model.setdefault(model, {})
            if vectorizer in model_results:
                raise ValueError(f"{model} modeli için tekrarlı {vectorizer} sonucu var.")
            metrics: dict[str, float] = {}
            for metric in CLASSIFICATION_METRICS:
                try:
                    value = float(comparison[metric])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"BoW/TF-IDF karşılaştırma CSV'sinde {row_number}. satırdaki {metric} değeri geçersiz."
                    ) from error
                if not 0 <= value <= 1:
                    raise ValueError(f"{metric} değeri 0 ile 1 arasında olmalıdır.")
                metrics[metric] = value
            model_results[vectorizer] = metrics

        if not bow_tfidf_by_model:
            raise ValueError("BoW/TF-IDF karşılaştırma CSV'sinde model sonucu bulunamadı.")
        for model, model_results in bow_tfidf_by_model.items():
            if set(model_results) != set(VECTORIZER_NAMES):
                raise ValueError(
                    f"{model} modelinde Bag of Words (BoW) ve TF-IDF sonuçlarının ikisi de bulunmalı."
                )

        parsed_terms: list[dict[str, Any]] = []
        corpus_stats: dict[str, Any] | None = None
        if bow_tfidf_term_rows is not None:
            for row_number, term_row in enumerate(bow_tfidf_term_rows, start=2):
                term = (term_row.get("term") or "").strip()
                if not term:
                    raise ValueError(
                        f"BoW/TF-IDF terim CSV'sinde {row_number}. satırda term alanı boş."
                    )
                try:
                    ngram_size = int(term_row["ngram_size"])
                    term_frequency = int(term_row["term_frequency"])
                    document_frequency = int(term_row["document_frequency"])
                    document_share = float(term_row["document_share"])
                    mean_tfidf = float(term_row["mean_tfidf"])
                    max_tfidf = float(term_row["max_tfidf"])
                    idf = float(term_row["idf"])
                    vocab_size = int(term_row["vocab_size"])
                    unigram_count = int(term_row["unigram_count"])
                    bigram_count = int(term_row["bigram_count"])
                    non_zero_count = int(term_row["non_zero_count"])
                    sparsity = float(term_row["sparsity"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"BoW/TF-IDF terim CSV'sinde {row_number}. satırın sayısal değerleri geçersiz."
                    ) from error
                if ngram_size not in (1, 2) or term_frequency < 1 or document_frequency < 1:
                    raise ValueError(
                        f"BoW/TF-IDF terim CSV'sinde {row_number}. satırın frekans/n-gram bilgisi geçersiz."
                    )
                if not (0 <= document_share <= 1 and 0 <= mean_tfidf <= 1 and 0 <= max_tfidf <= 1 and 0 <= sparsity <= 1):
                    raise ValueError(
                        f"BoW/TF-IDF terim CSV'sinde {row_number}. satırın oran/TF-IDF değerleri 0 ile 1 arasında olmalı."
                    )
                if corpus_stats is None:
                    corpus_stats = {
                        "vocab_size": vocab_size,
                        "unigram_count": unigram_count,
                        "bigram_count": bigram_count,
                        "non_zero_count": non_zero_count,
                        "sparsity": sparsity,
                    }
                parsed_terms.append(
                    {
                        "term": term,
                        "ngram_size": ngram_size,
                        "term_frequency": term_frequency,
                        "document_frequency": document_frequency,
                        "document_share": document_share,
                        "mean_tfidf": mean_tfidf,
                        "max_tfidf": max_tfidf,
                        "idf": idf,
                    }
                )

        test_count = round(len(review_rows) * 0.2)
        bow_tfidf_comparison = {
            "models": [
                {
                    "model": model,
                    "bow": model_results["Bag of Words (BoW)"],
                    "tfidf": model_results["TF-IDF"],
                    "delta_accuracy": (
                        model_results["TF-IDF"]["test_accuracy"]
                        - model_results["Bag of Words (BoW)"]["test_accuracy"]
                    ),
                    "delta_f1": (
                        model_results["TF-IDF"]["test_f1"]
                        - model_results["Bag of Words (BoW)"]["test_f1"]
                    ),
                    "delta_roc_auc": (
                        model_results["TF-IDF"]["test_roc_auc"]
                        - model_results["Bag of Words (BoW)"]["test_roc_auc"]
                    ),
                }
                for model, model_results in bow_tfidf_by_model.items()
            ],
            "row_count": len(bow_tfidf_rows),
            "train_count": len(review_rows) - test_count,
            "test_count": test_count,
            "label_source": "Yorum metninden türetilen sözlük etiketleri",
            "corpus_stats": corpus_stats,
            "terms": parsed_terms,
            "top_bow_terms": sorted(
                parsed_terms,
                key=lambda item: (item["term_frequency"], item["mean_tfidf"]),
                reverse=True,
            )[:12],
            "top_tfidf_terms": sorted(
                parsed_terms,
                key=lambda item: (item["mean_tfidf"], item["term_frequency"]),
                reverse=True,
            )[:12],
            "top_bigram_terms": [
                item for item in parsed_terms if item["ngram_size"] == 2
            ][:10],
        }

    if smote_rows is not None:
        smote_by_model: dict[str, dict[str, dict[str, float]]] = {}
        class_distribution: dict[str, Any] | None = None
        for row_number, comparison in enumerate(smote_rows, start=2):
            model = (comparison.get("model") or "").strip()
            smote_status = comparison.get("smote_status")
            if not model or smote_status not in SMOTE_CONDITIONS:
                raise ValueError(
                    f"SMOTE karşılaştırma CSV'sinde {row_number}. satırın model/SMOTE durumu geçersiz."
                )
            model_results = smote_by_model.setdefault(model, {})
            if smote_status in model_results:
                raise ValueError(f"{model} modeli için tekrarlı {smote_status} sonucu var.")
            metrics: dict[str, float] = {}
            for metric in CLASSIFICATION_METRICS:
                try:
                    value = float(comparison[metric])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"SMOTE karşılaştırma CSV'sinde {row_number}. satırdaki {metric} değeri geçersiz."
                    ) from error
                if not 0 <= value <= 1:
                    raise ValueError(f"{metric} değeri 0 ile 1 arasında olmalıdır.")
                metrics[metric] = value
            model_results[smote_status] = metrics

            if class_distribution is None and "orig_train_count" in comparison:
                try:
                    orig_train_count = int(comparison["orig_train_count"])
                    resampled_train_count = int(comparison["resampled_train_count"])
                    synthetic_count = int(comparison["synthetic_count"])
                    test_count_val = int(comparison["test_count"])
                    orig_pos = int(comparison["orig_pos_count"])
                    orig_neu = int(comparison["orig_neu_count"])
                    orig_neg = int(comparison["orig_neg_count"])
                    res_pos = int(comparison["resampled_pos_count"])
                    res_neu = int(comparison["resampled_neu_count"])
                    res_neg = int(comparison["resampled_neg_count"])
                    test_pos = int(comparison.get("test_pos_count", 0))
                    test_neu = int(comparison.get("test_neu_count", 0))
                    test_neg = int(comparison.get("test_neg_count", 0))
                    k_neighbors = int(comparison.get("k_neighbors", 5))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"SMOTE karşılaştırma CSV'sinde {row_number}. satırın sınıf dağılım adetleri geçersiz."
                    ) from error
                class_distribution = {
                    "orig_train_count": orig_train_count,
                    "resampled_train_count": resampled_train_count,
                    "synthetic_count": synthetic_count,
                    "test_count": test_count_val,
                    "k_neighbors": k_neighbors,
                    "before": {
                        "olumlu": orig_pos,
                        "nötr": orig_neu,
                        "olumsuz": orig_neg,
                    },
                    "after": {
                        "olumlu": res_pos,
                        "nötr": res_neu,
                        "olumsuz": res_neg,
                    },
                    "synthetic_by_class": {
                        "olumlu": res_pos - orig_pos,
                        "nötr": res_neu - orig_neu,
                        "olumsuz": res_neg - orig_neg,
                    },
                    "test": {
                        "olumlu": test_pos,
                        "nötr": test_neu,
                        "olumsuz": test_neg,
                    },
                }

        if not smote_by_model:
            raise ValueError("SMOTE karşılaştırma CSV'sinde model sonucu bulunamadı.")
        for model, model_results in smote_by_model.items():
            if set(model_results) != set(SMOTE_CONDITIONS):
                raise ValueError(
                    f"{model} modelinde SMOTE yok ve SMOTE var sonuçlarının ikisi de bulunmalı."
                )

        default_test_count = round(len(review_rows) * 0.2)
        smote_comparison = {
            "models": [
                {
                    "model": model,
                    "without_smote": model_results["SMOTE yok"],
                    "with_smote": model_results["SMOTE var"],
                    "delta_accuracy": (
                        model_results["SMOTE var"]["test_accuracy"]
                        - model_results["SMOTE yok"]["test_accuracy"]
                    ),
                    "delta_precision": (
                        model_results["SMOTE var"]["test_precision"]
                        - model_results["SMOTE yok"]["test_precision"]
                    ),
                    "delta_recall": (
                        model_results["SMOTE var"]["test_recall"]
                        - model_results["SMOTE yok"]["test_recall"]
                    ),
                    "delta_f1": (
                        model_results["SMOTE var"]["test_f1"]
                        - model_results["SMOTE yok"]["test_f1"]
                    ),
                    "delta_roc_auc": (
                        model_results["SMOTE var"]["test_roc_auc"]
                        - model_results["SMOTE yok"]["test_roc_auc"]
                    ),
                }
                for model, model_results in smote_by_model.items()
            ],
            "row_count": len(smote_rows),
            "train_count": (
                class_distribution["orig_train_count"]
                if class_distribution
                else len(review_rows) - default_test_count
            ),
            "resampled_train_count": (
                class_distribution["resampled_train_count"]
                if class_distribution
                else None
            ),
            "synthetic_count": (
                class_distribution["synthetic_count"]
                if class_distribution
                else None
            ),
            "test_count": (
                class_distribution["test_count"]
                if class_distribution
                else default_test_count
            ),
            "k_neighbors": (
                class_distribution["k_neighbors"] if class_distribution else 5
            ),
            "class_distribution": class_distribution,
            "label_source": "Yorum metninden türetilen sözlük etiketleri",
        }

    if training_size_rows is not None:
        results_by_size: dict[float, dict[str, Any]] = {}
        for row_number, comparison in enumerate(training_size_rows, start=2):
            try:
                training_size = float(comparison["training_size"])
                train_count = int(comparison["train_count"])
                test_count = int(comparison["test_count"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Eğitim oranı karşılaştırma CSV'sinde {row_number}. satırın oranı/adetleri geçersiz."
                ) from error
            if training_size not in TRAINING_SIZE_PROPORTIONS:
                raise ValueError(
                    f"Eğitim oranı karşılaştırma CSV'sinde {row_number}. satırın eğitim oranı desteklenmiyor."
                )
            if train_count + test_count != len(review_rows):
                raise ValueError(
                    f"Eğitim oranı karşılaştırma CSV'sinde {row_number}. satırın eğitim/test adetleri yorum sayısıyla uyuşmuyor."
                )
            split = results_by_size.setdefault(
                training_size,
                {
                    "training_size": training_size,
                    "train_count": train_count,
                    "test_count": test_count,
                    "models": {},
                },
            )
            if split["train_count"] != train_count or split["test_count"] != test_count:
                raise ValueError("Eğitim oranı karşılaştırma CSV'sinde aynı oran için adetler tutarsız.")
            model = (comparison.get("model") or "").strip()
            if model not in CLASSIFICATION_MODEL_NAMES or model in split["models"]:
                raise ValueError(
                    f"Eğitim oranı karşılaştırma CSV'sinde {row_number}. satırın model adı geçersiz veya tekrarlı."
                )
            metrics: dict[str, float] = {}
            for metric in CLASSIFICATION_METRICS:
                try:
                    value = float(comparison[metric])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Eğitim oranı karşılaştırma CSV'sinde {row_number}. satırdaki {metric} değeri geçersiz."
                    ) from error
                if not 0 <= value <= 1:
                    raise ValueError(f"{metric} değeri 0 ile 1 arasında olmalıdır.")
                metrics[metric] = value
            split["models"][model] = metrics

        if set(results_by_size) != set(TRAINING_SIZE_PROPORTIONS):
            raise ValueError("Eğitim oranı karşılaştırma CSV'sinde altı eğitim oranının tamamı bulunmalı.")
        for training_size, split in results_by_size.items():
            if set(split["models"]) != set(CLASSIFICATION_MODEL_NAMES):
                raise ValueError(
                    f"%{training_size:.0%} eğitim oranında dört klasik modelin tamamı bulunmalı."
                )
            split["models"] = [
                {"model": model, **split["models"][model]}
                for model in CLASSIFICATION_MODEL_NAMES
            ]
        training_size_comparison = {
            "splits": [
                results_by_size[training_size]
                for training_size in TRAINING_SIZE_PROPORTIONS
            ],
            "label_source": "Yorum metninden türetilen sözlük etiketleri",
        }

    if aspect_rows is not None:
        if len(aspect_rows) != len(review_rows):
            raise ValueError(
                "Sıfır-atış analiz CSV'sindeki satır sayısı yorum CSV'siyle eşleşmiyor; analizi yeniden çalıştırın."
            )
        seen_numbers: set[int] = set()
        for analysis in aspect_rows:
            try:
                review_number = int(analysis["review_number"])
                threshold = float(analysis["threshold"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Sıfır-atış analiz CSV'sinde geçersiz satır veya eşik var.") from error
            if review_number < 1 or review_number > len(review_rows) or review_number in seen_numbers:
                raise ValueError("Sıfır-atış analiz CSV'sinde geçersiz/tekrarlı review_number var.")
            seen_numbers.add(review_number)
            if not 0 < threshold < 1:
                raise ValueError("Sıfır-atış analiz CSV'sinde eşik 0 ile 1 arasında olmalıdır.")
            if aspect_threshold is not None and threshold != aspect_threshold:
                raise ValueError("Sıfır-atış analiz CSV'sindeki eşikler tutarsız.")
            aspect_threshold = threshold
            model = (analysis.get("model") or "").strip()
            if not model or (aspect_model is not None and model != aspect_model):
                raise ValueError("Sıfır-atış analiz CSV'sindeki model bilgisi boş veya tutarsız.")
            aspect_model = model
            analysis_status = analysis.get("analysis_status")
            if analysis_status == "eligible":
                eligible_count += 1
            elif analysis_status == "short_review":
                short_review_count += 1
            else:
                raise ValueError("Sıfır-atış analiz CSV'sinde bilinmeyen analiz durumu var.")

            review_aspects = []
            for aspect_id, label, _ in ASPECTS:
                sentiment = analysis.get(f"{aspect_id}_sentiment")
                if sentiment not in {"positive", "negative", "mixed", "not_mentioned", "short_review"}:
                    raise ValueError(f"Sıfır-atış analiz CSV'sinde {aspect_id} etiketi geçersiz.")
                scores: dict[str, float | None] = {}
                for polarity in ("positive", "negative"):
                    raw_score = (analysis.get(f"{aspect_id}_{polarity}_score") or "").strip()
                    try:
                        score = float(raw_score) if raw_score else None
                    except ValueError as error:
                        raise ValueError(f"Sıfır-atış analiz CSV'sinde {aspect_id} puanı sayı değil.") from error
                    if score is not None and not 0 <= score <= 1:
                        raise ValueError(f"Sıfır-atış analiz CSV'sinde {aspect_id} puanı 0 ile 1 arasında olmalı.")
                    scores[polarity] = score
                if analysis_status == "eligible" and (
                    scores["positive"] is None or scores["negative"] is None
                ):
                    raise ValueError(f"Analiz edilen {aspect_id} hedefinde model puanı eksik.")
                if analysis_status == "short_review" and any(
                    score is not None for score in scores.values()
                ):
                    raise ValueError("Kısa yorumlarda model puanı bulunmamalı.")
                if analysis_status == "eligible":
                    aspect_counts[aspect_id][sentiment] += 1
                review_aspects.append(
                    {
                        "id": aspect_id,
                        "label": label,
                        "sentiment": sentiment,
                        "positive_score": scores["positive"],
                        "negative_score": scores["negative"],
                    }
                )
            aspect_reviews[review_number] = {
                "analysis_status": analysis_status,
                "aspects": review_aspects,
            }

        if seen_numbers != set(range(1, len(review_rows) + 1)):
            raise ValueError("Sıfır-atış analiz CSV'sinde yorum sıra numaraları eksik.")

    for review_number, row in enumerate(review_rows, start=1):
        review_text = (row.get("review") or "").strip()
        if not review_text:
            raise ValueError("Yorum CSV'sinde boş bir review alanı bulundu.")
        rating = parse_rating(row.get("rating") or "")
        sentiment = sentiment_label(review_text)
        month = parse_review_month(row.get("date") or "")
        if rating is not None:
            parsed_ratings.append(rating)
            ratings[int(rating)] += 1
        if month is not None:
            months[month] += 1
        sentiments[sentiment] += 1
        reviews.append(
            {
                "reviewer": (row.get("reviewer") or "").strip() or "Anonim",
                "date": (row.get("date") or "").strip(),
                "rating": rating,
                "title": (row.get("title") or "").strip(),
                "review": review_text,
                "sentiment": sentiment,
                "analysis_status": aspect_reviews.get(review_number, {}).get(
                    "analysis_status", "unavailable"
                ),
                "aspects": aspect_reviews.get(review_number, {}).get("aspects", []),
            }
        )

    total = len(reviews)
    aspect_analysis = None
    if aspect_rows is not None:
        aspect_analysis = {
            "model": aspect_model,
            "threshold": aspect_threshold,
            "eligible_count": eligible_count,
            "short_review_count": short_review_count,
            "aspects": [
                {
                    "id": aspect_id,
                    "label": label,
                    "positive_count": aspect_counts[aspect_id]["positive"],
                    "negative_count": aspect_counts[aspect_id]["negative"],
                    "mixed_count": aspect_counts[aspect_id]["mixed"],
                    "not_mentioned_count": aspect_counts[aspect_id]["not_mentioned"],
                }
                for aspect_id, label, _ in ASPECTS
            ],
        }
    return {
        "summary": {
            "review_count": total,
            "average_rating": (
                round(sum(parsed_ratings) / len(parsed_ratings), 2)
                if parsed_ratings
                else None
            ),
            "positive_share": (
                round(sentiments["olumlu"] / total * 100, 1) if total else 0
            ),
            "rated_count": len(parsed_ratings),
        },
        "ratings": [
            {"rating": score, "count": ratings[score]} for score in range(5, 0, -1)
        ],
        "sentiments": [
            {"sentiment": label, "count": sentiments[label]}
            for label in ("olumlu", "nötr", "olumsuz")
        ],
        "months": [
            {"month": month, "count": months[month]} for month in sorted(months)
        ],
        "models": [
            {"model": model, **values} for model, values in score_rows
        ],
        "metric_labels": [{"key": key, "label": label} for key, label in METRICS],
        "aspect_analysis": aspect_analysis,
        "stemming_comparison": stemming_comparison,
        "training_size_comparison": training_size_comparison,
        "bow_tfidf_comparison": bow_tfidf_comparison,
        "smote_comparison": smote_comparison,
        "reviews": reviews,
    }


def load_dashboard_data(
    reviews_path: Path,
    metrics_path: Path,
    aspect_analysis_path: Path = DEFAULT_ASPECT_ANALYSIS,
    stemming_comparison_path: Path = DEFAULT_STEMMING_COMPARISON,
    training_size_comparison_path: Path = DEFAULT_TRAINING_SIZE_COMPARISON,
    bow_tfidf_comparison_path: Path = DEFAULT_BOW_TFIDF_COMPARISON,
    bow_tfidf_terms_path: Path = DEFAULT_BOW_TFIDF_TERMS,
    smote_comparison_path: Path = DEFAULT_SMOTE_COMPARISON,
) -> dict[str, Any]:
    if not reviews_path.is_file():
        raise FileNotFoundError(f"Yorum CSV dosyası bulunamadı: {reviews_path}")
    with reviews_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"review", "rating", "date"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("Yorum CSV'sinde review, rating ve date sütunları olmalı.")
        review_rows = list(reader)
    aspect_rows = None
    if aspect_analysis_path.is_file():
        with aspect_analysis_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required_fields = set(analysis_fields())
            if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
                raise ValueError(
                    "Sıfır-atış analiz CSV'sinde gerekli sütunlar eksik; analizi yeniden çalıştırın."
                )
            aspect_rows = list(reader)
    stemming_rows = None
    if stemming_comparison_path.is_file():
        with stemming_comparison_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required_fields = {
                "preprocessing",
                "model",
                *CLASSIFICATION_METRICS,
            }
            if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
                raise ValueError(
                    "Stemming karşılaştırma CSV'sinde sütunlar eksik; karşılaştırmayı yeniden çalıştırın."
                )
            stemming_rows = list(reader)
    training_size_rows = None
    if training_size_comparison_path.is_file():
        with training_size_comparison_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required_fields = {
                "training_size",
                "train_count",
                "test_count",
                "model",
                *CLASSIFICATION_METRICS,
            }
            if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
                raise ValueError(
                    "Eğitim oranı karşılaştırma CSV'sinde sütunlar eksik; karşılaştırmayı yeniden çalıştırın."
                )
            training_size_rows = list(reader)
    bow_tfidf_rows = None
    if bow_tfidf_comparison_path.is_file():
        with bow_tfidf_comparison_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required_fields = {
                "vectorizer",
                "model",
                *CLASSIFICATION_METRICS,
            }
            if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
                raise ValueError(
                    "BoW/TF-IDF karşılaştırma CSV'sinde sütunlar eksik; karşılaştırmayı yeniden çalıştırın."
                )
            bow_tfidf_rows = list(reader)
    bow_tfidf_term_rows = None
    if bow_tfidf_terms_path.is_file():
        with bow_tfidf_terms_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required_fields = set(BOW_TFIDF_TERM_FIELDS)
            if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
                raise ValueError(
                    "BoW/TF-IDF terim CSV'sinde sütunlar eksik; karşılaştırmayı yeniden çalıştırın."
                )
            bow_tfidf_term_rows = list(reader)
    smote_rows = None
    if smote_comparison_path.is_file():
        with smote_comparison_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required_fields = {
                "smote_status",
                "model",
                *CLASSIFICATION_METRICS,
            }
            if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
                raise ValueError(
                    "SMOTE karşılaştırma CSV'sinde sütunlar eksik; karşılaştırmayı yeniden çalıştırın."
                )
            smote_rows = list(reader)
    return build_dashboard_data(
        review_rows,
        read_scores(metrics_path),
        aspect_rows,
        stemming_rows,
        training_size_rows,
        bow_tfidf_rows,
        bow_tfidf_term_rows,
        smote_rows,
    )


def create_handler(reviews_path: Path, metrics_path: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request_path = urlparse(self.path).path
            if request_path == "/":
                page_path = PROJECT_DIR / "dashboard.html"
                if not page_path.is_file():
                    self.send_error(500, "Internal Server Error", "dashboard.html bulunamadı.")
                    return
                content = page_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            if request_path == "/api/dashboard":
                try:
                    payload = load_dashboard_data(reviews_path, metrics_path)
                except (OSError, ValueError) as error:
                    self.log_error("Dashboard verisi yüklenemedi: %s", error)
                    payload = {"error": str(error)}
                    content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            self.send_error(404, "Not Found", "Sayfa bulunamadı.")

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="iPhone 15 yorum grafik panelini açar.")
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS, help="İşlenmiş yorum CSV'si.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="Model sonuçları CSV'si.")
    parser.add_argument("--host", default="127.0.0.1", help="Sunucu adresi (varsayılan: localhost).")
    parser.add_argument("--port", type=int, default=8000, help="Sunucu portu (varsayılan: 8000).")
    args = parser.parse_args()
    load_dashboard_data(args.reviews, args.metrics)
    server = ThreadingHTTPServer(
        (args.host, args.port), create_handler(args.reviews, args.metrics)
    )
    print(f"Grafik paneli hazır: http://{args.host}:{args.port}")
    print("Kapatmak için Ctrl+C tuşlarına basın.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nYerel grafik paneli kapatılıyor.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
