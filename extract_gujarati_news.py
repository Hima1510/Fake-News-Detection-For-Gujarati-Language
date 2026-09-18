"""
Extract Gujarati news (with fake/real labels) from:
  1. A "mixed language" dataset (multiple languages, needs Gujarati filtering)
  2. A "particular" (Gujarati-only) dataset

Combine both into a single JSON-L file where each line is:
    {"news": "<gujarati text>", "label": "fake" | "real"}

NOTE: Column names below are guesses (text/news, label, language).
Update COLUMN MAPPINGS section once you share the real file structure.
"""

import json
import re
import pandas as pd

# ----------------------------
# CONFIG — EDIT THESE PATHS
# ----------------------------
MIXED_DATASET_PATH = "/mnt/user-data/uploads/mixed_dataset.csv"       # multi-language dataset
GUJARATI_DATASET_PATH = "/mnt/user-data/uploads/gujarati_dataset.csv"  # gujarati-only dataset
OUTPUT_PATH = "/mnt/user-data/outputs/gujarati_news_combined.jsonl"

# Column name guesses — change to match your actual files
MIXED_TEXT_COL = "text"
MIXED_LABEL_COL = "label"
MIXED_LANG_COL = "language"   # set to None if there's no language column and you want script-detection instead

GUJARATI_TEXT_COL = "text"
GUJARATI_LABEL_COL = "label"

# Label normalization: map whatever values appear in your data to "fake"/"real"
LABEL_MAP = {
    "fake": "fake", "0": "fake", 0: "fake", "false": "fake",
    "real": "real", "1": "real", 1: "real", "true": "real",
}

# Gujarati Unicode block: U+0A80–U+0AFF
GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")


def is_gujarati_text(text: str, threshold: float = 0.3) -> bool:
    """Detect Gujarati text by checking the proportion of Gujarati-script characters."""
    if not isinstance(text, str) or not text.strip():
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    guj_count = sum(1 for ch in letters if GUJARATI_RE.match(ch))
    return (guj_count / len(letters)) >= threshold


def normalize_label(raw_label):
    key = str(raw_label).strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    if raw_label in LABEL_MAP:
        return LABEL_MAP[raw_label]
    return None  # unrecognized label -> will be dropped


def load_dataset(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json"):
        return pd.read_json(path)
    elif path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    elif path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def extract_from_mixed(df):
    records = []
    for _, row in df.iterrows():
        text = row.get(MIXED_TEXT_COL)

        # Filter to Gujarati rows: prefer an explicit language column if present
        if MIXED_LANG_COL and MIXED_LANG_COL in df.columns:
            lang_val = str(row.get(MIXED_LANG_COL, "")).strip().lower()
            if lang_val not in ("gu", "gujarati", "guj"):
                continue
        else:
            if not is_gujarati_text(text):
                continue

        label = normalize_label(row.get(MIXED_LABEL_COL))
        if label is None or not isinstance(text, str) or not text.strip():
            continue

        records.append({"news": text.strip(), "label": label})
    return records


def extract_from_gujarati_only(df):
    records = []
    for _, row in df.iterrows():
        text = row.get(GUJARATI_TEXT_COL)
        label = normalize_label(row.get(GUJARATI_LABEL_COL))
        if label is None or not isinstance(text, str) or not text.strip():
            continue
        records.append({"news": text.strip(), "label": label})
    return records


def main():
    mixed_df = load_dataset(MIXED_DATASET_PATH)
    guj_df = load_dataset(GUJARATI_DATASET_PATH)

    mixed_records = extract_from_mixed(mixed_df)
    guj_records = extract_from_gujarati_only(guj_df)

    print(f"Extracted {len(mixed_records)} Gujarati rows from mixed dataset")
    print(f"Extracted {len(guj_records)} rows from Gujarati-only dataset")

    combined = mixed_records + guj_records

    # Optional: drop exact duplicate news text
    seen = set()
    deduped = []
    for rec in combined:
        if rec["news"] not in seen:
            seen.add(rec["news"])
            deduped.append(rec)

    print(f"Total combined: {len(combined)} | after de-dup: {len(deduped)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
