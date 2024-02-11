""" This script identifies the top n keywords in a document using the TF-IDF method. """

import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf_on_background_data(corpus: list[str]) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on the background data."""
    print("Fitting TF-IDF vectorizer on background data. This may take a minute")

    s = time.time()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    e = time.time()
    print(f"TF-IDF vectorizer fit in {e - s:.2f} seconds")
    return vectorizer


def find_top_n_words_in_document(
    vectorizer: TfidfVectorizer, document: list[str], n: int
) -> list[str]:
    """Find the top n words in a document using the TF-IDF method.

    The top words are determined by the sum of the TF-IDF scores for each word in the document.
    """
    print(f"Finding top {n} keywords in document")
    tf_idf_scores = vectorizer.transform(document)

    tf_idf_per_word = pd.DataFrame(
        tf_idf_scores.toarray(), columns=vectorizer.get_feature_names_out()
    ).sum(axis=0)

    return tf_idf_per_word.nlargest(n).index.tolist()


def read_in_background_data(
    background_path: str,
) -> list[str]:
    """Read in the background data assume to be a CSV file with a column named 'TEXT'."""

    print(f"Reading in background data from {background_path}")
    data = pd.read_csv(background_path)
    print(f"Read in {len(data):,} rows of background data")

    if "TEXT" not in data.columns:
        raise ValueError(
            f"Background data must have a column named 'TEXT'. Columns found: {data.columns}"
        )

    return data["TEXT"].tolist()


def read_in_document(document_path: str) -> list[str]:
    """Read in the document to find keywords in."""
    with open(document_path, "r") as file:
        return file.readlines()


def find_keywords(
    background_path: str = "data/background_tf_idf.csv",
    document_path: str = "data/dictation_long.txt",
    n_keywords: int = 25,
) -> list[str]:
    """Find the top n keywords in a document using the TF-IDF method.

    :param background_path: Path to the background data. This should be a CSV file with a column named 'TEXT'.
    :param document_path: Path to the document to find keywords in. This should be a text file with one sentence per line.
    :param n_keywords: Number of keywords to find in the document."""

    background_data = read_in_background_data(background_path)
    document = read_in_document(document_path)
    vectorizer = fit_tfidf_on_background_data(background_data)
    return find_top_n_words_in_document(vectorizer, document, n_keywords)


def main(
    output_path: str = "data/keywords.txt",
    background_path: str = "data/background_tf_idf.csv",
    document_path: str = "data/dictation_long.txt",
    n_keywords: int = 25,
):
    keywords = find_keywords()

    print(f"Writing keywords to {output_path}")
    with open(output_path, "w") as file:
        for keyword in keywords:
            file.write(f"{keyword}\n")


if __name__ == "__main__":
    main()
