"""Text representation: Bag of Words dan TF-IDF dari hasil preprocessing.

Dibangun dari kolom tokens_stemmed_final dan tokens_lemmatized_final
(keluaran preprocessing.py), masing-masing sejajar supaya bisa dibandingkan.

Pemakaian:
    python text_representation.py [--input data/comments_preprocessed.csv] [--out-dir data]
        [--ngram-max 1]

--ngram-max 1 menghasilkan unigram saja (default, nama file tanpa akhiran).
--ngram-max 2 menghasilkan unigram+bigram (nama file berakhiran "_bigram"),
supaya bisa dibandingkan berdampingan dengan hasil unigram.
"""

import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

META_COLUMNS = ["post_url", "username", "comment_text", "timestamp", "likes_count"]


def tokens_from_pipe_string(text):
    return [t for t in text.split("|") if t]


def build_matrix(vectorizer_cls, texts, ngram_max):
    vectorizer = vectorizer_cls(
        tokenizer=tokens_from_pipe_string,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, ngram_max),
    )
    matrix = vectorizer.fit_transform(texts)
    df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df, vectorizer


def save_representation(df, token_column, out_dir, label, ngram_max):
    texts = df[token_column].fillna("")
    meta = df[META_COLUMNS].reset_index(drop=True)
    suffix = "" if ngram_max == 1 else "_bigram"

    bow_df, _ = build_matrix(CountVectorizer, texts, ngram_max)
    tfidf_df, tfidf_vectorizer = build_matrix(TfidfVectorizer, texts, ngram_max)

    bow_out = pd.concat([meta, bow_df], axis=1)
    tfidf_out = pd.concat([meta, tfidf_df], axis=1)

    bow_path = f"{out_dir}/bow_{label}{suffix}.csv"
    tfidf_path = f"{out_dir}/tfidf_{label}{suffix}.csv"
    vectorizer_path = f"{out_dir}/tfidf_{label}{suffix}_vectorizer.joblib"
    bow_out.to_csv(bow_path, index=False)
    tfidf_out.to_csv(tfidf_path, index=False)
    joblib.dump(tfidf_vectorizer, vectorizer_path)

    print(f"[{label}{suffix}] vocab: {len(bow_df.columns)} kata")
    print(f"  Bag of Words -> {bow_path}")
    print(f"  TF-IDF       -> {tfidf_path}")
    print(f"  TF-IDF vectorizer (untuk VSM) -> {vectorizer_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/comments_preprocessed.csv")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--ngram-max", type=int, default=1,
                         help="1 = unigram saja, 2 = unigram+bigram")
    args = parser.parse_args()

    df = pd.read_csv(args.input, keep_default_na=False)

    save_representation(df, "tokens_stemmed_final", args.out_dir, "stemmed", args.ngram_max)
    save_representation(df, "tokens_lemmatized_final", args.out_dir, "lemmatized", args.ngram_max)


if __name__ == "__main__":
    main()
