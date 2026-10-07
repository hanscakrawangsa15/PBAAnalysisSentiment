"""Scrape komentar TikTok lewat Apify actor clockworks/tiktok-comments-scraper.

Pemakaian:
    python scrape_comments.py [--posts posts.txt] [--limit 500] [--out data/comments.csv]

Daftar URL video diambil dari posts.txt (satu URL per baris, baris '#' diabaikan).
Hasil di-append ke satu CSV gabungan supaya komentar dari beberapa video bisa
dikumpulkan untuk tahap analisis NLP berikutnya.
"""

import argparse
import csv
import os
import sys

from apify_client import ApifyClient
from dotenv import load_dotenv

ACTOR_ID = "clockworks/tiktok-comments-scraper"
CSV_FIELDS = ["post_url", "username", "comment_text", "timestamp", "likes_count"]


def load_post_urls(path):
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def scrape_post(client, post_url, limit):
    run_input = {"postURLs": [post_url], "commentsPerPost": limit, "maxRepliesPerComment": 0}
    run = client.actor(ACTOR_ID).call(run_input=run_input)
    items = client.dataset(run.default_dataset_id).list_items().items

    rows = []
    for item in items:
        rows.append({
            "post_url": item.get("videoWebUrl", post_url),
            "username": item.get("uniqueId", ""),
            "comment_text": item.get("text", ""),
            "timestamp": item.get("createTimeISO", ""),
            "likes_count": item.get("diggCount", 0),
        })
    return rows


def append_rows_to_csv(path, rows):
    file_exists = os.path.isfile(path)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", default="posts.txt", help="File berisi daftar URL video TikTok")
    parser.add_argument("--limit", type=int, default=500, help="Jumlah maksimum komentar per post")
    parser.add_argument("--out", default="data/comments.csv", help="Path CSV output (di-append)")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        sys.exit("APIFY_API_TOKEN tidak ditemukan. Isi file .env terlebih dahulu.")

    post_urls = load_post_urls(args.posts)
    if not post_urls:
        sys.exit(f"Tidak ada URL di {args.posts}. Tambahkan minimal satu URL video TikTok.")

    client = ApifyClient(token)

    total_rows = 0
    for post_url in post_urls:
        print(f"Scraping komentar: {post_url}")
        try:
            rows = scrape_post(client, post_url, args.limit)
        except Exception as e:
            print(f"  Gagal scrape {post_url}: {e}", file=sys.stderr)
            continue

        append_rows_to_csv(args.out, rows)
        total_rows += len(rows)
        print(f"  Dapat {len(rows)} komentar")

    print(f"Selesai. Total {total_rows} komentar ditulis ke {args.out}")


if __name__ == "__main__":
    main()
