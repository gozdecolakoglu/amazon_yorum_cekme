import argparse
import csv
from html import escape
from pathlib import Path


METRICS = (
    ("train_accuracy", "Eğitim Doğruluğu"),
    ("test_accuracy", "Test Doğruluğu"),
    ("test_precision", "Kesinlik (Precision)"),
    ("test_recall", "Duyarlılık (Recall)"),
    ("test_f1", "F1 Skoru"),
    ("test_roc_auc", "ROC-AUC"),
)
COLORS = ("#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2", "#be123c")
MODEL_LABELS = {
    "Lojistik Regresyon": "Lojistik Reg.",
    "Support Vector Machine": "SVM",
    "Türkçe BERT (dbmdz/bert-base-turkish-cased)": "Türkçe BERT",
}


def read_scores(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"model", *(key for key, _ in METRICS)}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV'de gerekli sütunlar eksik: {', '.join(sorted(missing))}")

        rows = []
        for line_number, record in enumerate(reader, start=2):
            model = (record.get("model") or "").strip()
            if not model:
                raise ValueError(f"{line_number}. satırda model adı boş.")
            values = {}
            for key, _ in METRICS:
                try:
                    value = float(record[key])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{line_number}. satırdaki {key} değeri sayı değil."
                    ) from error
                if not 0 <= value <= 1:
                    raise ValueError(
                        f"{line_number}. satırdaki {key} değeri 0 ile 1 arasında olmalı."
                    )
                values[key] = value
            rows.append((model, values))

    if not rows:
        raise ValueError("CSV'de çizilecek model sonucu bulunamadı.")
    return rows


def chart_svg(rows):
    width = 1260
    columns = 3
    panel_width = 390
    panel_height = 95 + len(rows) * 26
    panel_gap = 18
    chart_top = 165
    row_gap = 20
    height = chart_top + 2 * panel_height + row_gap + 48
    plot_width = 200
    plot_offset = 115
    colors = [COLORS[index % len(COLORS)] for index in range(len(rows))]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="chart-title chart-description">',
        "<title id=\"chart-title\">Sınıflandırma modellerinin başarı ölçütleri</title>",
        '<desc id="chart-description">Altı ölçütteki model sonuçları, 0 ile 100 '
        "yüzde ölçeğinde yatay çubuk grafiklerle karşılaştırılmıştır.</desc>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="36" y="45" font-family="Arial, sans-serif" font-size="27" '
        'font-weight="700" fill="#0f172a">Sınıflandırma Modelleri: Ölçüt Karşılaştırması</text>',
        '<text x="36" y="75" font-family="Arial, sans-serif" font-size="15" '
        'fill="#475569">classification_accuracy.csv • Tüm değerler yüzde olarak (0–100)</text>',
    ]

    legend_start = 36
    legend_step = 245
    for index, (model, _) in enumerate(rows):
        x = legend_start + index * legend_step
        label = escape(model)
        parts.append(
            f'<circle cx="{x}" cy="119" r="7" fill="{colors[index]}"/>'
            f'<text x="{x + 14}" y="124" font-family="Arial, sans-serif" '
            f'font-size="13" fill="#334155">{label}</text>'
        )

    for metric_index, (key, title) in enumerate(METRICS):
        column = metric_index % columns
        row_index = metric_index // columns
        x = 30 + column * (panel_width + panel_gap)
        y = chart_top + row_index * (panel_height + row_gap)
        plot_x = x + plot_offset
        axis_y = y + 50 + len(rows) * 26
        parts.append(
            f'<rect x="{x}" y="{y}" width="{panel_width}" height="{panel_height}" '
            'rx="10" fill="#ffffff" stroke="#cbd5e1"/>'
            f'<text x="{x + 16}" y="{y + 29}" font-family="Arial, sans-serif" '
            f'font-size="17" font-weight="700" fill="#0f172a">{escape(title)}</text>'
        )
        for tick in (0, 50, 100):
            tick_x = plot_x + plot_width * tick / 100
            parts.append(
                f'<line x1="{tick_x}" y1="{y + 42}" x2="{tick_x}" y2="{axis_y}" '
                'stroke="#e2e8f0"/>'
                f'<text x="{tick_x}" y="{axis_y + 17}" text-anchor="middle" '
                'font-family="Arial, sans-serif" font-size="11" fill="#64748b">'
                f"{tick}%</text>"
            )

        for model_index, (model, values) in enumerate(rows):
            bar_y = y + 50 + model_index * 26
            percentage = values[key] * 100
            label = escape(MODEL_LABELS.get(model, model))
            parts.append(
                f'<text x="{x + 10}" y="{bar_y + 13}" font-family="Arial, sans-serif" '
                f'font-size="12" fill="#334155">{label}</text>'
                f'<rect x="{plot_x}" y="{bar_y}" width="{plot_width}" height="16" '
                'rx="4" fill="#f1f5f9"/>'
                f'<rect x="{plot_x}" y="{bar_y}" width="{plot_width * values[key]:.2f}" '
                f'height="16" rx="4" fill="{colors[model_index]}"/>'
                f'<text x="{plot_x + plot_width + 6}" y="{bar_y + 13}" '
                'font-family="Arial, sans-serif" font-size="12" font-weight="600" '
                f'fill="#0f172a">{percentage:.1f}%</text>'
            )

    parts.extend(
        [
            f'<text x="36" y="{height - 19}" font-family="Arial, sans-serif" '
            'font-size="12" fill="#64748b">Değerler CSV’deki metriklerden üretilmiştir; '
            "eğitim doğruluğu ve test ölçütleri ayrı gösterilir.</text>",
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="classification_accuracy.csv verilerinden karşılaştırma grafiği üretir."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "classification_accuracy.csv",
        help="Metriklerin bulunduğu CSV dosyası.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "classification_accuracy.svg",
        help="Oluşturulacak SVG grafik dosyası.",
    )
    args = parser.parse_args()
    rows = read_scores(args.input)
    args.output.write_text(chart_svg(rows), encoding="utf-8")
    print(f"Grafik oluşturuldu: {args.output}")


if __name__ == "__main__":
    main()
