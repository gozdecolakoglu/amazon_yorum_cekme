"""Export the local dashboard data as a static JSON file for Vercel."""

from __future__ import annotations

import json
from pathlib import Path

from dashboard import DEFAULT_METRICS, DEFAULT_REVIEWS, load_dashboard_data


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "dashboard-data.json"


def main() -> None:
    data = load_dashboard_data(DEFAULT_REVIEWS, DEFAULT_METRICS)
    OUTPUT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Vercel için dashboard verisi oluşturuldu: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
