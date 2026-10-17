"""İşlenmiş Amazon iPhone 15 yorumlarıyla duygu sınıflandırıcılarını eğitir."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# Girdi ve çıktı yolları, betiğin ek argüman verilmeden proje klasöründe çalışmasını sağlar.
DEFAULT_INPUT = Path("iphone15_reviews_processed.csv")
DEFAULT_OUTPUT = Path("classification_accuracy.csv")
DEFAULT_BERT_MODEL = "dbmdz/bert-base-turkish-cased"
# Etiket sırası BERT sınıf numaralarıyla eşleştiğinden bütün BERT işlemlerinde sabit tutulur.
LABELS = ("olumsuz", "nötr", "olumlu")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
# Türkçe karakterleri ASCII'ye indirgemek sözlükte tek tip kök önekleriyle eşleşmeyi sağlar.
TRANSLATION = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
)
# Kök önekleri çekimli biçimleri de yakalar; örneğin "beğendim" sözcüğü "beg" ile başlar.
POSITIVE_STEMS = (
    "basar",
    "beg",
    "guzel",
    "harika",
    "hizli",
    "iyi",
    "kalite",
    "memnun",
    "mutlu",
    "orijinal",
    "orjinal",
    "ozel",
    "saglam",
    "sorunsuz",
    "super",
    "tesekkur",
    "uygun",
)
NEGATIVE_STEMS = (
    "berbat",
    "bozuk",
    "calismiyor",
    "eksik",
    "gecik",
    "iptal",
    "iade",
    "kotu",
    "memnuniyetsiz",
    "olmadi",
    "pahali",
    "rezalet",
    "sorun",
    "yapalmadi",
    "yapilmadi",
)
# Olumsuzluk sözcüğü, hemen öncesindeki duygu belirten kökün kutbunu tersine çevirir.
NEGATORS = ("degil", "yok")


def sentiment_label(review_text: str) -> str:
    """Yorumdaki olumlu/olumsuz kökleri sayarak duygu etiketini belirler."""
    # Küçük harfe çevirme öncesinde birleşik noktalı İ'yi sadeleştirerek düzenli token üretir.
    normalized = review_text.lower().replace("i\u0307", "i").translate(TRANSLATION)
    tokens = TOKEN_PATTERN.findall(normalized)
    positive_score = 0
    negative_score = 0

    # En uzun önek önce seçilir; böylece "sorunsuz" sözcüğü "sorun" diye olumsuz sayılmaz.
    positive_stems = sorted(POSITIVE_STEMS, key=len, reverse=True)
    negative_stems = sorted(NEGATIVE_STEMS, key=len, reverse=True)
    for index, token in enumerate(tokens):
        positive_match = next((stem for stem in positive_stems if token.startswith(stem)), None)
        negative_match = next((stem for stem in negative_stems if token.startswith(stem)), None)
        if positive_match is None and negative_match is None:
            continue

        # "İyi değil" ve "sorun yok" gibi ifadelerde sonraki olumsuzluk sözcüğü kutbu çevirir.
        is_negated = index + 1 < len(tokens) and tokens[index + 1] in NEGATORS
        if positive_match is not None and (
            negative_match is None or len(positive_match) >= len(negative_match)
        ):
            if is_negated:
                negative_score += 1
            else:
                positive_score += 1
        elif is_negated:
            positive_score += 1
        else:
            negative_score += 1

    # Eşleşme yoksa veya iki kutbun sayısı eşitse yorum belirsiz kabul edilip nötr etiketlenir.
    if positive_score == negative_score:
        return "nötr"
    return "olumlu" if positive_score > negative_score else "olumsuz"


def load_classification_data(path: Path) -> tuple[list[str], list[str]]:
    """CSV'den yorum metinlerini yükler ve duygu etiketlerini yorumdan türetir."""
    if not path.is_file():
        raise FileNotFoundError(f"Processed review CSV does not exist: {path}")

    texts: list[str] = []
    labels: list[str] = []
    # utf-8-sig, Excel'in ekleyebildiği BOM işaretini başlık adından temizler.
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        # Artık yıldız puanı kullanılmaz; etiketler yalnızca yorum cümlelerinden çıkarılır.
        if not reader.fieldnames or "review" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'review' column.")

        for row_number, row in enumerate(reader, start=2):
            text = (row.get("review") or "").strip()
            # Hatalı satırı atlamak yerine numarasıyla bildirmek veri sorununu görünür kılar.
            if not text:
                raise ValueError(f"Review text is empty at CSV row {row_number}.")
            texts.append(text)
            labels.append(sentiment_label(text))

    if not texts:
        raise ValueError(f"No review rows were found in {path}.")
    # Metin sözlüğü yalnızca tek bir sınıf bulduysa sınıflandırıcı eğitilemez.
    if len(set(labels)) < 2:
        raise ValueError("Yorum sözlüğü en az iki duygu sınıfı üretmelidir.")
    return texts, labels


def split_data(
    texts: Sequence[str],
    labels: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Sınıf oranlarını koruyan, tekrar üretilebilir eğitim/test ayrımı oluşturur."""
    # Stratify, azınlık duygu sınıflarının iki kümeye de dağılmasına yardımcı olur.
    x_train, x_test, y_train, y_test = train_test_split(
        list(texts),
        list(labels),
        test_size=test_size,
        random_state=random_state,
        stratify=list(labels),
    )
    return x_train, x_test, y_train, y_test


def _accuracy_result(
    model_name: str,
    model: object,
    x_train: object,
    y_train: Sequence[str],
    x_test: object,
    y_test: Sequence[str],
) -> dict[str, str | float]:
    # Eğitim ve test başarıları ayrı ölçülerek ezberleme ile genelleme farkı görülebilir.
    train_predictions = model.predict(x_train)  # type: ignore[attr-defined]
    test_predictions = model.predict(x_test)  # type: ignore[attr-defined]
    return {
        "model": model_name,
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
    }


def run_classical_models(
    x_train: Sequence[str],
    y_train: Sequence[str],
    x_test: Sequence[str],
    y_test: Sequence[str],
) -> list[dict[str, str | float]]:
    """TF-IDF özellikleriyle Naive Bayes, lojistik regresyon, SVM ve Random Forest eğitir."""
    # N-gramlar tek sözcüklerin yanında "çok iyi" gibi kısa ifadeleri de yakalar.
    # Vektörleştirici yalnızca eğitim metinlerine uydurulur; test bilgisinin sızması engellenir.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, sublinear_tf=True)
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    models = (
        # Dört algoritmaya aynı özellikleri vermek, ölçümleri yöntemler arasında karşılaştırır.
        ("Naive Bayes", MultinomialNB()),
        ("Lojistik Regresyon", LogisticRegression(max_iter=2_000, random_state=42)),
        ("Support Vector Machine", LinearSVC(random_state=42)),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    )
    results: list[dict[str, str | float]] = []
    for name, model in models:
        model.fit(train_vectors, y_train)
        results.append(
            _accuracy_result(name, model, train_vectors, y_train, test_vectors, y_test)
        )
    return results


def run_turkish_bert(
    x_train: Sequence[str],
    y_train: Sequence[str],
    x_test: Sequence[str],
    y_test: Sequence[str],
    model_name: str = DEFAULT_BERT_MODEL,
    epochs: int = 3,
) -> dict[str, str | float]:
    """Türkçe BERT'i ince ayarlar ve eğitim/test doğruluğunu ölçer."""
    # Ağır bağımlılıklar yalnızca BERT çalıştırılırken yüklenir; klasik modelleri etkilemez.
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Sabit tohum, ağırlık başlangıcını ve eğitim sırasını mümkün olduğunca tekrarlanabilir yapar.
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Etiket numaraları, modelin çıktılarının Türkçe duygu adlarıyla eşleştirilmesini sağlar.
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    ).to(device)

    def encode(texts: Sequence[str], labels: Sequence[str]) -> TensorDataset:
        # Kesme uzun metinleri sınırlar; dolgu aynı gruptaki tensörlerin aynı boyutta olmasını sağlar.
        tokens = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded_labels = torch.tensor([label_to_id[label] for label in labels])
        return TensorDataset(tokens["input_ids"], tokens["attention_mask"], encoded_labels)

    # BERT yalnızca eğitim kümesini görür; test kümesi son değerlendirmeye ayrılır.
    training_data = encode(x_train, y_train)
    training_loader = DataLoader(
        training_data,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    # Her mini grupta gradyanları hesaplayıp ağırlıkları güncellemek ince ayar döngüsüdür.
    for _ in range(epochs):
        for input_ids, attention_mask, labels in training_loader:
            optimizer.zero_grad()
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            output.loss.backward()
            optimizer.step()

    def predict(texts: Sequence[str], labels: Sequence[str]) -> list[str]:
        dataset = encode(texts, labels)
        predictions: list[str] = []
        # Değerlendirme sırasında gradyan hesaplamamak bellek ve çalışma süresini azaltır.
        model.eval()
        with torch.inference_mode():
            for input_ids, attention_mask, _ in DataLoader(dataset, batch_size=8):
                output = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                )
                predictions.extend(
                    LABELS[index] for index in output.logits.argmax(dim=1).cpu().tolist()
                )
        return predictions

    train_predictions = predict(x_train, y_train)
    test_predictions = predict(x_test, y_test)
    return {
        "model": f"Türkçe BERT ({model_name})",
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
    }


def write_results(
    results: Sequence[dict[str, str | float]],
    output_path: Path,
) -> None:
    # Üst klasör yoksa oluşturulur; sonuçlar Excel uyumlu BOM'lu UTF-8 CSV olarak yazılır.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("model", "train_accuracy", "test_accuracy"),
        )
        writer.writeheader()
        writer.writerows(results)


def print_results(results: Sequence[dict[str, str | float]]) -> None:
    # Yüzde biçimi, CSV'deki 0–1 oranlarını okunabilir doğruluk değerlerine çevirir.
    print(f"{'Model':<42} {'Eğitim doğruluğu':>18} {'Test doğruluğu':>16}")
    for result in results:
        print(
            f"{result['model']:<42} "
            f"{result['train_accuracy']:>17.2%} "
            f"{result['test_accuracy']:>15.2%}"
        )


def main() -> None:
    # Argümanlar, farklı veri/model dosyalarının kodu değiştirmeden seçilmesini sağlar.
    parser = argparse.ArgumentParser(
        description="Train sentiment classifiers on processed Amazon reviews."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Processed reviews CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Accuracy results CSV.")
    parser.add_argument(
        "--bert-model",
        default=DEFAULT_BERT_MODEL,
        help="Hugging Face Turkish BERT model identifier or local model directory.",
    )
    args = parser.parse_args()

    texts, labels = load_classification_data(args.input)
    x_train, x_test, y_train, y_test = split_data(texts, labels)
    # Etiket adetleri özellikle dengesiz veri kümelerinde doğruluğu yorumlamaya yardım eder.
    print(f"Yorum: {len(texts)} | Eğitim: {len(x_train)} | Test: {len(x_test)}")
    print("Etiket dağılımı:", ", ".join(f"{label}={labels.count(label)}" for label in LABELS))

    results = run_classical_models(x_train, y_train, x_test, y_test)
    # BERT indirme/eğitimi uzun sürebileceği için klasik sonuçlar önce kalıcılaştırılır.
    write_results(results, args.output)
    print_results(results)

    results.append(run_turkish_bert(x_train, y_train, x_test, y_test, args.bert_model))
    write_results(results, args.output)
    print_results(results)
    print(f"\nDoğruluk sonuçları kaydedildi: {args.output}")


if __name__ == "__main__":
    # Dosya doğrudan çalıştırıldığında eğitimi başlatır; içe aktarmada yan etki oluşturmaz.
    main()
