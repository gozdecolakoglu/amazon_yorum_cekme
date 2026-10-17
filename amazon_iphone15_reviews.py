"""Kaydedilmiş Amazon HTML veya CSV dosyalarından iPhone 15 yorumlarını çıkarıp işler."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import types
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol, Sequence

from bs4 import BeautifulSoup, Tag
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from TurkishStemmer import TurkishStemmer

if TYPE_CHECKING:
    from zemberek import TurkishMorphology


# Protokoller, Zemberek ve kök bulucu gibi bağımlılıkları testlerde taklit edilebilir kılar.
class Stemmer(Protocol):
    def stem(self, word: str) -> str: ...


class Lemma(Protocol):
    lemma: str


class AnalysisResult(Protocol):
    item: Lemma

    def is_unknown(self) -> bool: ...


class WordAnalysis(Protocol):
    analysis_results: Sequence[AnalysisResult]


class Morphology(Protocol):
    def analyze(self, word: str) -> WordAnalysis: ...


LOGGER = logging.getLogger(__name__)
# Varsayılanlar, komut satırında yol verilmediğinde kullanılacak dosyaları belirler.
DEFAULT_SOURCE_DIR = Path.home() / "Desktop" / "amazon_yorumlar_files"
DEFAULT_SOURCE_FILE = DEFAULT_SOURCE_DIR / "amazon_yorumları_cekilenler.csv"
DEFAULT_OUTPUT_FILE = Path("iphone15_reviews_processed.csv")
REVIEW_SELECTORS = ('[data-hook="review"]', ".review")
REVIEW_BODY_SELECTORS = ('[data-hook="review-body"]', ".review-text")
TOKENIZER = RegexpTokenizer(r"[a-zçğıöşü0-9]+")
# Negatif bakış, iPhone 15 Pro/Max/Plus yorumlarının temel model verisine karışmasını önler.
IPHONE_15_PATTERN = re.compile(r"\biphone\s*15\b(?!\s*(?:pro|max|plus)\b)", re.IGNORECASE)
# NLTK verisi bulunamadığında da Türkçe stop-word temizliği yapılabilsin diye yerel liste tutulur.
TURKISH_STOP_WORDS = {
    "acaba", "ama", "ancak", "artık", "aslında", "az", "bazı", "belki", "ben",
    "benden", "beni", "benim", "beri", "beş", "bile", "bir", "birçok", "biri",
    "birkaç", "biz", "bize", "bizi", "bizim", "böyle", "bu", "buna", "bunda",
    "bundan", "bunlar", "bunları", "bunların", "bunu", "bunun", "burada",
    "çok", "çünkü", "da", "daha", "de", "defa", "diğer", "diye", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "için", "kadar", "ki",
    "kim", "mı", "mi", "mu", "mü", "nasıl", "ne", "neden", "nerde", "nerede",
    "nereye", "niçin", "niye", "o", "olan", "olarak", "oldu", "olduğu",
    "olduklarını", "olmak", "olması", "olmuş", "on", "ona", "ondan", "onlar",
    "onlardan", "onları", "onların", "onu", "onun", "orada", "oysa", "öyle",
    "şey", "şu", "şuna", "şunda", "şundan", "şunları", "şunu", "tarafından",
    " üç", "ve", "veya", "ya", "yani",
}
CSV_FIELDS = (
    "review_id",
    "reviewer",
    "date",
    "rating",
    "title",
    "review",
    "lowercase",
    "punctuation_cleaned",
    "tokens",
    "stopword_removed_tokens",
    "stemmed_tokens",
    "single_character_removed_tokens",
    "lemmatized_tokens",
)


def load_stop_words() -> set[str]:
    """Varsa NLTK'nin Türkçe listesini yerel stop-word listesiyle birleştirir."""
    words = TURKISH_STOP_WORDS.copy()
    # Başında boşluk bulunan hatalı/istenmeyen öğeyi, normalleştirilmiş kümeye geçmeden çıkarır.
    words.discard(" üç")
    try:
        words.update(stopwords.words("turkish"))
    except LookupError:
        LOGGER.warning(
            "NLTK Turkish stopwords corpus is unavailable; using the bundled Turkish list."
        )
    return {word.strip().lower() for word in words}


def create_morphology() -> TurkishMorphology:
    """Python 3.13+ ortamında antlr4 uyumluluk köprüsü kurup Zemberek'i yükler."""
    # Antlr4 4.8 eski typing.io yolunu bekler; yeni Python'da bulunmadığında uyumluluk modülü eklenir.
    if not hasattr(typing, "io"):
        typing_io = types.ModuleType("typing.io")
        typing_io.TextIO = typing.TextIO
        sys.modules["typing.io"] = typing_io
        setattr(typing, "io", typing_io)

    from zemberek import TurkishMorphology

    return TurkishMorphology.create_with_defaults()


def _select_text(element: Tag, selectors: Iterable[str]) -> str:
    # Birden fazla seçici, Amazon HTML'indeki değişen sınıf ve data-hook biçimlerini destekler.
    for selector in selectors:
        found = element.select_one(selector)
        if found is not None:
            return found.get_text(" ", strip=True)
    return ""


def _review_nodes(soup: BeautifulSoup) -> list[Tag]:
    # Önce Amazon'un kararlı data-hook niteliği, yoksa alternatif CSS sınıfı denenir.
    nodes = soup.select(REVIEW_SELECTORS[0])
    if not nodes:
        nodes = soup.select(REVIEW_SELECTORS[1])
    return nodes


def _is_iphone_15_page(soup: BeautifulSoup) -> bool:
    # Yorum metinleri ürün adı aramasını kirletebileceği için kopya HTML'den çıkarılır.
    page_copy = BeautifulSoup(str(soup), "html.parser")
    for review in _review_nodes(page_copy):
        review.decompose()

    product_context = [page_copy.get_text(" ", strip=True)]
    product_title = page_copy.select_one("#productTitle, .product-title")
    if product_title is not None:
        product_context.append(product_title.get_text(" ", strip=True))
    product_context.extend(
        meta.get("content", "")
        for meta in page_copy.select('meta[property="og:title"], meta[name="title"]')
    )
    return bool(IPHONE_15_PATTERN.search(" ".join(product_context)))


def extract_reviews(source_dir: Path) -> list[dict[str, str]]:
    """Kaydedilmiş HTML sayfalarından iPhone 15 yorum kayıtlarını okur."""
    # Tek dosya veya klasör girişi kabul etmek, tek sayfa ve toplu işlem kullanımını sağlar.
    if source_dir.is_file():
        html_files = [source_dir]
    elif source_dir.is_dir():
        html_files = sorted((*source_dir.glob("*.html"), *source_dir.glob("*.htm")))
    else:
        raise FileNotFoundError(f"Review source does not exist: {source_dir}")

    reviews: list[dict[str, str]] = []
    # Aynı yorum birden çok kaydedilmiş sayfada bulunabileceği için kimlikleri tekilleştirir.
    seen_review_ids: set[str] = set()

    for html_file in html_files:
        # read_bytes, BeautifulSoup'un sayfanın karakter kodlamasını saptamasını sağlar.
        soup = BeautifulSoup(html_file.read_bytes(), "html.parser")
        review_nodes = _review_nodes(soup)
        if not review_nodes or not _is_iphone_15_page(soup):
            continue

        for node in review_nodes:
            body = _select_text(node, REVIEW_BODY_SELECTORS)
            if not body:
                continue

            review_id = str(node.get("id", "")).strip()
            title = _select_text(node, ('[data-hook="review-title"]', ".review-title"))
            # Kimliği olmayan kayıtlar kaybedilmez; kimliği olan yinelenenler atlanır.
            if review_id and review_id in seen_review_ids:
                continue

            if review_id:
                seen_review_ids.add(review_id)
            reviews.append(
                {
                    "review_id": review_id,
                    "reviewer": _select_text(
                        node, ('[data-hook="review-author"]', ".review-author")
                    ),
                    "date": _select_text(
                        node, ('[data-hook="review-date"]', ".review-date")
                    ),
                    "rating": _select_text(
                        node, ('[data-hook="review-star-rating"]', ".review-rating")
                    ),
                    "title": title,
                    "review": body,
                }
            )

    if not reviews:
        raise ValueError(
            f"No iPhone 15 reviews were found in saved HTML source {source_dir}. "
            "Save the Amazon reviews page itself as HTML, including its review content."
        )
    return reviews


def _normalize_header(header: str) -> str:
    # Türkçe başlıkları alan takma adlarıyla eşleştirmek için noktalama ve Türkçe i varyantları sadeleştirilir.
    normalized = header.strip().lower().replace("ı", "i").replace("i\u0307", "i")
    return re.sub(r"[^a-z0-9]", "", normalized)


def extract_reviews_from_csv(source_file: Path) -> list[dict[str, str]]:
    """CSV'den yorumları okur ve ürün bilgisi varsa yalnızca iPhone 15'i seçer."""
    if not source_file.is_file():
        raise FileNotFoundError(f"Review CSV does not exist: {source_file}")

    review_headers = (
        "reviewtext",
        "reviewbody",
        "review",
        "comment",
        "content",
        "text",
        "yorummetni",
        "yorumicerigi",
        "yorumlar",
        "yorum",
    )
    product_headers = {
        "model", "product", "productmodel", "productname", "producttitle",
        "urun", "urunadi", "urunmodeli", "urunbasligi",
    }
    field_aliases = {
        "review_id": ("reviewid", "id", "yorumid"),
        "reviewer": ("reviewer", "author", "username", "kullanici", "yorumcu"),
        "date": ("reviewdate", "date", "tarih"),
        "rating": ("ratingvalue", "rating", "stars", "star", "puan"),
        "title": ("reviewtitle", "yorumbasligi", "baslik", "title"),
    }

    with source_file.open("r", encoding="utf-8-sig", newline="") as input_file:
        # DictReader başlıkları satır değerlerine bağlar; utf-8-sig Excel BOM'unu temizler.
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError(f"Review CSV has no header row: {source_file}")

        headers = {
            _normalize_header(header): header
            for header in reader.fieldnames
            if header is not None
        }
        # Yaygın İngilizce/Türkçe sütun adları arasından ilk eşleşen yorum sütunu seçilir.
        review_header = next(
            (headers[name] for name in review_headers if name in headers),
            None,
        )
        if review_header is None:
            raise ValueError(
                "Could not identify a review text column in the CSV. "
                "Use a column such as review, reviewText, comment, or yorum."
            )

        product_columns = [
            header
            for header in reader.fieldnames
            if header is not None and _normalize_header(header) in product_headers
        ]
        # Bulunmayan isteğe bağlı alanlar boş bırakılır; yorum metni ve puan gibi veriler korunur.
        output_columns = {
            output_name: next(
                (
                    headers[name]
                    for name in aliases
                    if name in headers
                ),
                "",
            )
            for output_name, aliases in field_aliases.items()
        }
        reviews: list[dict[str, str]] = []
        for row in reader:
            review_text = (row.get(review_header) or "").strip()
            # Boş yorumlar model eğitimine anlamlı metin sağlamadığından veri kümesine alınmaz.
            if not review_text:
                continue

            if product_columns:
                # Ürün sütunu varsa Pro/Max/Plus gibi başka modelleri hariç tut.
                product_text = " ".join(row.get(column, "") or "" for column in product_columns)
                if not IPHONE_15_PATTERN.search(product_text):
                    continue

            review = {
                name: (row.get(column) or "").strip() if column else ""
                for name, column in output_columns.items()
            }
            review["review"] = review_text
            reviews.append(review)

    if not reviews:
        raise ValueError(
            f"No usable iPhone 15 reviews were found in CSV {source_file}. "
            "Check that it contains review text and, for mixed products, an "
            "identifiable product/model column."
        )
    return reviews


def preprocess_review(
    text: str,
    morphology: Morphology,
    stop_words: set[str],
    stemmer: Stemmer | None = None,
) -> dict[str, str]:
    """Yorumu küçük harf, noktalama, token, stop-word, kök ve lemma aşamalarından geçirir."""
    # Türkçe büyük I/İ dönüşümü, standart lower() çağrısının tek başına oluşturabileceği farkları düzeltir.
    lowercase = text.translate(str.maketrans({"İ": "i", "I": "ı"})).lower()
    punctuation_cleaned = re.sub(
        r"[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s]", " ", lowercase
    )
    punctuation_cleaned = re.sub(r"\s+", " ", punctuation_cleaned).strip()
    tokens = TOKENIZER.tokenize(punctuation_cleaned)
    # Stop-word temizliği içerik taşımayan sık sözcükleri sonraki aşamalardan çıkarır.
    stopword_removed = [word for word in tokens if word not in stop_words]

    # Enjekte edilen kök bulucu testlerde deterministik taklit sağlar; yoksa gerçek Türkçe kökleyici kullanılır.
    active_stemmer = stemmer or TurkishStemmer()
    stemmed = [active_stemmer.stem(word) for word in stopword_removed]
    multi_character = [word for word in stopword_removed if len(word) > 1]
    lemmatized: list[str] = []
    for word in multi_character:
        # Zemberek bilinmeyen sözcükte güvenilir lemma vermeyebilir; böyle durumda özgün sözcük korunur.
        analysis = morphology.analyze(word)
        if analysis.analysis_results and not analysis.analysis_results[0].is_unknown():
            lemmatized.append(analysis.analysis_results[0].item.lemma)
        else:
            lemmatized.append(word)

    return {
        "lowercase": lowercase,
        "punctuation_cleaned": punctuation_cleaned,
        "tokens": " ".join(tokens),
        "stopword_removed_tokens": " ".join(stopword_removed),
        "stemmed_tokens": " ".join(stemmed),
        "single_character_removed_tokens": " ".join(multi_character),
        "lemmatized_tokens": " ".join(lemmatized),
    }


def write_reviews_csv(
    reviews: list[dict[str, str]],
    output_path: Path,
    morphology: Morphology,
    stop_words: set[str],
) -> None:
    """Çıkarılan alanları ve her metin ön işleme aşamasını CSV'ye kaydeder."""
    # Ara klasörlerin önceden var olması gerekmez; BOM Excel'in Türkçe metni doğru açmasına yardım eder.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for review in reviews:
            writer.writerow(
                {
                    **review,
                    **preprocess_review(review["review"], morphology, stop_words),
                }
            )


def main() -> None:
    # Argüman ayrıştırıcı, dosya/klasör ve çıktı yollarının çalıştırma anında özelleştirilmesini sağlar.
    parser = argparse.ArgumentParser(
        description="Extract and preprocess iPhone 15 reviews from an Amazon CSV or HTML."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SOURCE_FILE,
        help="Input review CSV, saved HTML file, or directory containing saved HTML pages.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Destination CSV file.",
    )
    args = parser.parse_args()

    try:
        # Uzantı CSV yolunu seçer; diğer girdiler tek HTML dosyası veya HTML klasörü kabul edilir.
        if args.input.suffix.lower() == ".csv":
            reviews = extract_reviews_from_csv(args.input)
        else:
            reviews = extract_reviews(args.input)
    except (FileNotFoundError, ValueError) as error:
        # argparse.error, kullanıcıya girdi hatasını anlaşılır biçimde bildirip geçersiz çıktıyı önler.
        parser.error(str(error))

    # Morfoloji motoru pahalı olduğundan yalnızca veri çıkarımı başarılı olduktan sonra başlatılır.
    morphology = create_morphology()
    write_reviews_csv(reviews, args.output, morphology, load_stop_words())
    print(f"{len(reviews)} iPhone 15 reviews saved to {args.output}")


if __name__ == "__main__":
    # Günlük yapılandırması ve ana akış yalnızca betik doğrudan çalıştırıldığında etkinleşir.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
