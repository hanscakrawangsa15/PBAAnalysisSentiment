"""Pipeline preprocessing teks komentar Bahasa Indonesia.

Menerapkan 10 teknik text preprocessing, masing-masing disimpan sebagai kolom
terpisah agar efek tiap teknik bisa diperiksa satu per satu:

 1. Lowering              -> text_lower
 2. Punctuation Removal   -> text_no_punct
 3. Expand Contractions   -> text_expanded      (normalisasi kata gaul/singkatan)
 4. Spelling Correction   -> text_corrected
 5. Tokenization          -> tokens_raw
 6. Stopword Removal      -> tokens_no_stopword
 7. Stemming              -> tokens_stemmed          (cabang A, dari tokens_no_stopword)
 8. Lemmatization         -> tokens_lemmatized       (cabang B, dari tokens_no_stopword)
 9. Common Words Removal  -> tokens_stemmed_no_common / tokens_lemmatized_no_common
10. Rare Words Removal    -> tokens_stemmed_final / tokens_lemmatized_final

Stemming dan lemmatization dijalankan sejajar dari hasil stopword removal yang sama
supaya hasil kedua teknik bisa dibandingkan, bukan ditumpuk berurutan.

Pemakaian:
    python preprocessing.py [--input data/comments.csv] [--output data/comments_preprocessed.csv]
        [--common-top-n 10] [--rare-min-freq 2]
"""

import argparse
import difflib
import string
from collections import Counter, defaultdict

import pandas as pd
from nltk.tokenize import word_tokenize
from wordfreq import top_n_list
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.stopword import StopWord

from slang_dict import SLANG_DICT

_stemmer = StemmerFactory().create_stemmer()
_lemmatizer = Lemmatizer()
_STOPWORD_SET = set(StopWord().get_stopword())

_VOCAB_LIST = top_n_list("id", 30000)
_VOCAB = set(_VOCAB_LIST)
_VOCAB_BY_FIRST_LETTER = defaultdict(list)
for _w in _VOCAB_LIST:
    if _w:
        _VOCAB_BY_FIRST_LETTER[_w[0]].append(_w)


# 1. Lowering
def lower_text(text):
    return text.lower()


# 2. Punctuation Removal
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


# 3. Expand Contractions (normalisasi kata gaul/singkatan)
def expand_contractions(text):
    words = text.split()
    return " ".join(SLANG_DICT.get(w, w) for w in words)


# 4. Spelling Correction
def _correct_word(word):
    if not word.isalpha() or len(word) <= 3 or word in _VOCAB:
        return word
    candidates = _VOCAB_BY_FIRST_LETTER.get(word[0], [])
    candidates = [c for c in candidates if abs(len(c) - len(word)) <= 2]
    matches = difflib.get_close_matches(word, candidates, n=1, cutoff=0.8)
    return matches[0] if matches else word


def correct_spelling(text):
    return " ".join(_correct_word(w) for w in text.split())


# 5. Tokenization
def tokenize(text):
    return word_tokenize(text)


# 6. Stopword Removal
def remove_stopwords(tokens):
    return [t for t in tokens if t not in _STOPWORD_SET]


# 7. Stemming
def stem_tokens(tokens):
    stemmed = (_stemmer.stem(t) for t in tokens)
    return [t for t in stemmed if t]


# 8. Lemmatization
def lemmatize_tokens(tokens):
    if not tokens:
        return []
    lemmatized = _lemmatizer.lemmatize(" ".join(tokens))
    return lemmatized.split() if lemmatized else []


# 9 & 10. Common / Rare Words Removal (butuh statistik frekuensi seluruh korpus)
def build_frequency(token_lists):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    return counter


def get_common_words(counter, top_n):
    return {word for word, _ in counter.most_common(top_n)}


def get_rare_words(counter, min_freq):
    return {word for word, freq in counter.items() if freq <= min_freq}


def remove_words(tokens, words_to_remove):
    return [t for t in tokens if t not in words_to_remove]


def process_dataframe(df, common_top_n, rare_min_freq):
    df = df.copy()
    df["comment_text"] = df["comment_text"].fillna("")

    df["text_lower"] = df["comment_text"].apply(lower_text)
    df["text_no_punct"] = df["text_lower"].apply(remove_punctuation)
    df["text_expanded"] = df["text_no_punct"].apply(expand_contractions)
    df["text_corrected"] = df["text_expanded"].apply(correct_spelling)
    df["tokens_raw"] = df["text_corrected"].apply(tokenize)
    df["tokens_no_stopword"] = df["tokens_raw"].apply(remove_stopwords)

    df["tokens_stemmed"] = df["tokens_no_stopword"].apply(stem_tokens)
    df["tokens_lemmatized"] = df["tokens_no_stopword"].apply(lemmatize_tokens)

    stemmed_freq = build_frequency(df["tokens_stemmed"])
    lemmatized_freq = build_frequency(df["tokens_lemmatized"])

    stemmed_common = get_common_words(stemmed_freq, common_top_n)
    stemmed_rare = get_rare_words(stemmed_freq, rare_min_freq)
    lemmatized_common = get_common_words(lemmatized_freq, common_top_n)
    lemmatized_rare = get_rare_words(lemmatized_freq, rare_min_freq)

    df["tokens_stemmed_no_common"] = df["tokens_stemmed"].apply(
        lambda t: remove_words(t, stemmed_common)
    )
    df["tokens_stemmed_final"] = df["tokens_stemmed_no_common"].apply(
        lambda t: remove_words(t, stemmed_rare)
    )

    df["tokens_lemmatized_no_common"] = df["tokens_lemmatized"].apply(
        lambda t: remove_words(t, lemmatized_common)
    )
    df["tokens_lemmatized_final"] = df["tokens_lemmatized_no_common"].apply(
        lambda t: remove_words(t, lemmatized_rare)
    )

    token_columns = [
        "tokens_raw", "tokens_no_stopword", "tokens_stemmed", "tokens_lemmatized",
        "tokens_stemmed_no_common", "tokens_stemmed_final",
        "tokens_lemmatized_no_common", "tokens_lemmatized_final",
    ]
    for col in token_columns:
        df[col] = df[col].apply(lambda tokens: "|".join(tokens))

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/comments.csv", help="Path CSV hasil scraping")
    parser.add_argument("--output", default="data/comments_preprocessed.csv", help="Path CSV output")
    parser.add_argument("--common-top-n", type=int, default=10,
                         help="Jumlah kata paling sering muncul yang dibuang (Common Words Removal)")
    parser.add_argument("--rare-min-freq", type=int, default=2,
                         help="Kata dengan frekuensi korpus <= nilai ini dibuang (Rare Words Removal)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = process_dataframe(df, args.common_top_n, args.rare_min_freq)
    result.to_csv(args.output, index=False)
    print(f"Selesai. {len(result)} baris diproses -> {args.output}")


if __name__ == "__main__":
    main()
