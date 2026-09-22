"""Vector Space Model (VSM) search: cari komentar paling relevan terhadap sebuah query.

Query diproses lewat pipeline preprocessing yang sama (lowering, punctuation removal,
expand contractions, spelling correction, tokenization, stopword removal, stemming),
lalu diproyeksikan ke ruang vektor TF-IDF yang sama dengan korpus komentar (dibangun
oleh text_representation.py). Relevansi diukur dengan cosine similarity.

Pemakaian:
    python vsm_search.py "prabowo pidato bagus" [--top-k 10]
    python vsm_search.py "prabowo pidato bagus" --data data/comments_preprocessed.csv \
        --vectorizer data/tfidf_stemmed_vectorizer.joblib
"""

import argparse

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import (
    correct_spelling,
    expand_contractions,
    lower_text,
    remove_punctuation,
    remove_stopwords,
    stem_tokens,
    tokenize,
)
# tokens_from_pipe_string harus tersedia di namespace __main__ ini karena
# joblib menyimpan fungsi tokenizer TfidfVectorizer dengan referensi ke
# modul tempat ia dijalankan (__main__ saat text_representation.py dieksekusi).
from text_representation import tokens_from_pipe_string  # noqa: F401


def preprocess_query(query):
    text = lower_text(query)
    text = remove_punctuation(text)
    text = expand_contractions(text)
    text = correct_spelling(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return "|".join(tokens)


def search(query, df, token_column, vectorizer, top_k):
    doc_texts = df[token_column].fillna("")
    doc_matrix = vectorizer.transform(doc_texts)

    query_text = preprocess_query(query)
    query_vector = vectorizer.transform([query_text])

    scores = cosine_similarity(query_vector, doc_matrix)[0]
    ranked = scores.argsort()[::-1][:top_k]

    results = df.iloc[ranked][["post_url", "username", "comment_text"]].copy()
    results["similarity"] = scores[ranked]
    return results[results["similarity"] > 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Kata kunci pencarian")
    parser.add_argument("--data", default="data/comments_preprocessed.csv")
    parser.add_argument("--vectorizer", default="data/tfidf_stemmed_vectorizer.joblib")
    parser.add_argument("--token-column", default="tokens_stemmed_final")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    vectorizer = joblib.load(args.vectorizer)

    results = search(args.query, df, args.token_column, vectorizer, args.top_k)

    if results.empty:
        print("Tidak ada komentar yang relevan dengan query ini.")
        return

    for _, row in results.iterrows():
        print(f"[{row['similarity']:.4f}] (@{row['username']}) {row['comment_text']}")


if __name__ == "__main__":
    main()
