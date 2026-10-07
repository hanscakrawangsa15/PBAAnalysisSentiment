"""Buang baris komentar yang isinya cuma emoji dan/atau spasi (tidak ada teks
bermakna sama sekali), supaya waktu labeling manual tidak kebuang untuk baris
yang jelas tidak bisa dilabel.

Pemakaian:
    python clean_comments.py --input data/comments_batch2.csv [--output data/comments_batch2.csv]
"""

import argparse

import emoji
import pandas as pd


def is_meaningless(text):
    stripped = emoji.replace_emoji(str(text), replace="").strip()
    return stripped == ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV hasil scraping yang mau dibersihkan")
    parser.add_argument("--output", default=None, help="Default: timpa --input")
    args = parser.parse_args()

    df = pd.read_csv(args.input, keep_default_na=False)
    before = len(df)

    mask_meaningless = df["comment_text"].apply(is_meaningless)
    removed = df[mask_meaningless]
    cleaned = df[~mask_meaningless]

    out_path = args.output or args.input
    cleaned.to_csv(out_path, index=False)

    print(f"Baris awal   : {before}")
    print(f"Dibuang      : {len(removed)} (emoji-only/spasi-only)")
    print(f"Sisa         : {len(cleaned)} -> {out_path}")


if __name__ == "__main__":
    main()
