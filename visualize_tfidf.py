"""Visualisasi kata dengan skor TF-IDF tertinggi (ranked bar chart).

Skor tiap kata dihitung sebagai jumlah (sum) nilai TF-IDF-nya di seluruh
dokumen/komentar, lalu diurutkan dari yang tertinggi ke terendah.

Pemakaian:
    python visualize_tfidf.py [--input data/tfidf_stemmed.csv] [--top-n 20]
        [--out data/tfidf_top_words.png]
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

META_COLUMNS = ["post_url", "username", "comment_text", "timestamp", "likes_count"]

SURFACE = "#fcfcfb"
BAR_COLOR = "#2a78d6"
GRIDLINE = "#e1e0d9"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"


def top_words_by_tfidf(df, top_n):
    vocab_cols = [c for c in df.columns if c not in META_COLUMNS]
    scores = df[vocab_cols].sum(axis=0).sort_values(ascending=False).head(top_n)
    return scores


def plot_top_words(scores, title, out_path):
    scores = scores.iloc[::-1]  # tertinggi di atas saat digambar horizontal

    fig, ax = plt.subplots(figsize=(9, max(4, len(scores) * 0.35)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.barh(scores.index, scores.values, color=BAR_COLOR, height=0.6, zorder=3)

    max_val = scores.max()
    for bar, value in zip(bars, scores.values):
        ax.text(
            value + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center", ha="left",
            fontsize=9, color=TEXT_SECONDARY,
        )

    ax.set_xlim(0, max_val * 1.12)
    ax.set_title(title, fontsize=13, color=TEXT_PRIMARY, loc="left", pad=14, fontweight="bold")
    ax.set_xlabel("Total skor TF-IDF", fontsize=10, color=TEXT_MUTED)

    ax.tick_params(axis="y", labelsize=10, colors=TEXT_PRIMARY, length=0)
    ax.tick_params(axis="x", labelsize=9, colors=TEXT_MUTED, length=0)

    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Chart tersimpan -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/tfidf_stemmed.csv")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out", default="data/tfidf_top_words.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input, keep_default_na=False)
    scores = top_words_by_tfidf(df, args.top_n)

    label = args.input.split("/")[-1].replace(".csv", "")
    title = f"{args.top_n} Kata dengan Skor TF-IDF Tertinggi ({label})"
    plot_top_words(scores, title, args.out)


if __name__ == "__main__":
    main()
