"""Extract Gujarati fake/real news from the datasets present in this workspace.

This project contains:
  - GujaratiBharatFake*.csv files
  - Gujarati_val.csv
  - Gujarati_fake_news/*.txt
  - Gujarati_real_news/*.txt

The script normalizes all labels to either "fake" or "real" and writes a JSONL
file with rows shaped like:
    {"news_id": "<unique id>", "text": "<gujarati text>", "label": "fake" | "real"}
"""

import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_DATASET_PATHS = sorted({path for path in BASE_DIR.glob("Gujarati*.csv")})
EXCEL_DATASET_PATHS = [BASE_DIR / "indianlabels_normalized.xlsx"]
DATASET_PATHS = CSV_DATASET_PATHS + EXCEL_DATASET_PATHS
TEXT_DATASET_DIRS = [
    (BASE_DIR / "Gujarati_fake_news", "fake"),
    (BASE_DIR / "Gujarati_real_news", "real"),
]
OUTPUT_PATH = BASE_DIR / "gujarati_fake_real_combined.jsonl"

TEXT_COL = "text"
LABEL_COL = "label"
ID_COL = "id"

LABEL_MAP = {
    "fake": "fake",
    "0": "fake",
    0: "fake",
    "false": "fake",
    "real": "real",
    "1": "real",
    1: "real",
    "true": "real",
}

GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")


def is_gujarati_text(text: str, threshold: float = 0.3) -> bool:
    """Check whether a text sample is primarily Gujarati."""
    if not isinstance(text, str) or not text.strip():
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    guj_count = sum(1 for ch in letters if GUJARATI_RE.match(ch))
    return (guj_count / len(letters)) >= threshold


def normalize_label(raw_label):
    if raw_label is None or pd.isna(raw_label):
        return None

    if isinstance(raw_label, str):
        key = raw_label.strip().lower()
        if key in LABEL_MAP:
            return LABEL_MAP[key]

    if raw_label in LABEL_MAP:
        return LABEL_MAP[raw_label]

    return None


def load_dataset(path):
    path = str(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".json"):
        return pd.read_json(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".xlsx"):
        if not Path(path).exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        return pd.read_excel(path, engine="openpyxl")
    if path.endswith(".xls"):
        if not Path(path).exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def extract_from_csv(df, source_tag):
    records = []
    text_col = TEXT_COL if TEXT_COL in df.columns else next((c for c in df.columns if "text" in c.lower()), None)
    label_col = LABEL_COL if LABEL_COL in df.columns else next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), None)

    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns in {source_tag}: {list(df.columns)}")

    for idx, row in df.iterrows():
        text = row.get(text_col)
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if not text.strip() or not is_gujarati_text(text):
            continue

        label = normalize_label(row.get(label_col))
        if label is None:
            continue

        if ID_COL in df.columns and pd.notna(row.get(ID_COL)):
            news_id = f"{source_tag}_{row.get(ID_COL)}"
        else:
            news_id = f"{source_tag}_{idx}"

        records.append({"news_id": str(news_id), "text": text.strip(), "label": label})

    return records


def extract_from_excel(df, source_tag):
    records = []
    text_col = next((c for c in df.columns if "statement" in c.lower() or "text" in c.lower() or "news" in c.lower()), None)
    label_col = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), None)
    lang_col = next((c for c in df.columns if "language" in c.lower() or "lang" in c.lower()), None)

    if text_col is None or label_col is None:
        raise ValueError(f"Could not find Statement/Label columns in {source_tag}: {list(df.columns)}")

    for idx, row in df.iterrows():
        if lang_col is not None:
            lang_value = str(row.get(lang_col, "")).strip().lower()
            if lang_value and lang_value != "gu":
                continue

        text = row.get(text_col)
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if not text.strip() or not is_gujarati_text(text):
            continue

        label = normalize_label(row.get(label_col))
        if label is None:
            continue

        records.append({
            "news_id": f"{source_tag}_{idx}",
            "text": text.strip(),
            "label": label,
        })

    return records


def extract_from_text_folders():
    records = []
    for folder, label_name in TEXT_DATASET_DIRS:
        if not folder.exists():
            continue

        for txt_path in sorted(folder.glob("*.txt")):
            text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text or not is_gujarati_text(text):
                continue

            records.append({
                "news_id": f"{label_name}_{txt_path.stem}",
                "text": text,
                "label": label_name,
            })

    return records


def main():
    combined = []

    for path in DATASET_PATHS:
        if not path.exists():
            print(f"[SKIPPED] Missing file: {path}")
            continue

        try:
            df = load_dataset(path)
            source_tag = path.stem
            if path.suffix.lower() in {".xlsx", ".xls"}:
                records = extract_from_excel(df, source_tag)
            else:
                records = extract_from_csv(df, source_tag)
            print(f"Extracted {len(records)} rows from {path.name}")
            combined.extend(records)
        except Exception as exc:
            print(f"[SKIPPED] Could not load {path}: {exc}")

    text_records = extract_from_text_folders()
    print(f"Extracted {len(text_records)} rows from Gujarati fake/real text folders")
    combined.extend(text_records)

    seen = set()
    deduped = []
    for rec in combined:
        key = rec["text"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    print(f"Total before dedupe: {len(combined)} | after dedupe: {len(deduped)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps({
                "news_id": rec["news_id"],
                "text": rec["text"],
                "label": rec["label"],
            }, ensure_ascii=False) + "\n")

    print(f"Saved combined dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
