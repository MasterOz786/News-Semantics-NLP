from __future__ import annotations

import html
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import emoji
import numpy as np
import pandas as pd
import regex as re2
import spacy
from gensim.models import Word2Vec
from nltk import download
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def ensure_nltk_resources() -> None:
    for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        try:
            download(resource, quiet=True)
        except Exception:
            pass


ensure_nltk_resources()


class DatasetLoader:
    @staticmethod
    def load_covid19_fakenews(root_dir: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
        path = Path(root_dir)
        files = sorted(path.glob("**/*COVID-19*.csv"))
        rows = []
        for file in files:
            lower = file.name.lower()
            if "fake" in lower or "claimfake" in lower:
                label = "fake"
            elif "real" in lower or "claimreal" in lower:
                label = "real"
            else:
                continue
            try:
                df = pd.read_csv(file, encoding="utf-8", engine="python")
            except Exception:
                df = pd.read_csv(file, encoding="latin1", engine="python")
            columns = [c.lower().strip() for c in df.columns]
            df.columns = columns
            text_fields = [col for col in columns if col in {"content", "title", "abstract", "newstitle"}]
            if not text_fields:
                continue
            for _, row in df.iterrows():
                parts = [str(row[col]) for col in text_fields if pd.notna(row.get(col))]
                text = " ".join(parts)
                if len(text.strip()) == 0:
                    continue
                rows.append({"text": text, "label": label, "source": str(file.name)})
        if sample_size is not None and len(rows) > sample_size:
            rows = random.sample(rows, sample_size)
        return pd.DataFrame(rows)


class TextCleaner:
    ROMAN_URDU_MAP: Dict[str, str] = {
        "hai": "is",
        "hain": "are",
        "nahi": "not",
        "ka": "of",
        "ke": "that",
        "ko": "to",
        "ye": "this",
        "woh": "that",
        "aur": "and",
        "tum": "you",
        "hum": "we",
        "mera": "my",
        "meri": "my",
        "kya": "what",
        "kyun": "why",
        "jab": "when",
        "liye": "for",
        "abhi": "now",
        "bahut": "very",
        "bohot": "very",
        "acha": "good",
        "achha": "good",
        "bura": "bad",
        "sb": "all",
        "sabka": "everyone",
        "sach": "truth",
        "jhoot": "false",
        "sab": "all",
    }

    HTML_TAG = re.compile(r"<[^>]+>")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
    HANDLE_PATTERN = re.compile(r"@\w+", flags=re.IGNORECASE)
    EMOJI_PATTERN = emoji.get_emoji_regexp()
    PUNCT_REPEATS = re.compile(r"([!?.]){2,}")
    NON_PRINTABLE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
    WORD_PATTERN = re.compile(r"\b[\w']+\b")

    @classmethod
    def _replace_roman_urdu(cls, text: str) -> str:
        def mapper(match: re.Match) -> str:
            token = match.group(0).lower()
            return cls.ROMAN_URDU_MAP.get(token, token)

        return re.sub(r"\b[aiouy]*[a-z]+\b", mapper, text, flags=re.IGNORECASE)

    @classmethod
    def clean_text(cls, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = html.unescape(text)
        text = cls.HTML_TAG.sub(" ", text)
        text = cls.URL_PATTERN.sub(" ", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = cls.HANDLE_PATTERN.sub(" ", text)
        text = cls.PUNCT_REPEATS.sub(r"\1", text)
        text = cls.EMOJI_PATTERN.sub(" ", text)
        text = cls.NON_PRINTABLE.sub(" ", text)
        text = cls._replace_roman_urdu(text)
        text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip().lower()
        return text

    @classmethod
    def audit_noise(cls, texts: Sequence[str], sample_size: int = 200) -> pd.DataFrame:
        sampled = random.sample(list(texts), min(sample_size, len(texts)))
        reports = []
        for text in sampled:
            before = str(text)
            after = cls.clean_text(before)
            reports.append(
                {
                    "html_tags": int(bool(cls.HTML_TAG.search(before))),
                    "urls": int(bool(cls.URL_PATTERN.search(before))),
                    "handles": int(bool(cls.HANDLE_PATTERN.search(before))),
                    "emojis": int(bool(cls.EMOJI_PATTERN.search(before))),
                    "repeated_punctuation": int(bool(cls.PUNCT_REPEATS.search(before))),
                    "roman_urdu_terms": sum(
                        1
                        for token in cls.WORD_PATTERN.findall(before)
                        if token.lower() in cls.ROMAN_URDU_MAP
                    ),
                    "before_length": len(before.split()),
                    "after_length": len(after.split()),
                }
            )
        return pd.DataFrame(reports)


class TokenizerComparer:
    CUSTOM_REGEX = re.compile(r"\b[\w']+\b|[\u0600-\u06FF]+|[!?.,;:-]", flags=re.UNICODE)

    def __init__(self, spacy_model: Optional[str] = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = spacy.blank("en")

    @staticmethod
    def tokenize_nltk(text: str) -> List[str]:
        return word_tokenize(text)

    def tokenize_spacy(self, text: str) -> List[str]:
        return [token.text for token in self.nlp(text)]

    @classmethod
    def tokenize_custom(cls, text: str) -> List[str]:
        return cls.CUSTOM_REGEX.findall(text)

    @staticmethod
    def _oov_rate(tokens: Sequence[str]) -> float:
        if len(tokens) == 0:
            return 0.0
        unknown = sum(1 for token in tokens if token.isalpha() and not wordnet.synsets(token.lower()))
        return unknown / len(tokens)

    @staticmethod
    def _contractions(tokens: Sequence[str]) -> int:
        return sum(1 for token in tokens if "'" in token)

    def compare(self, texts: Sequence[str], sample_size: int = 50) -> pd.DataFrame:
        sampled = random.sample(list(texts), min(sample_size, len(texts)))
        results = []
        methods = ["nltk", "spacy", "custom"]
        for method in methods:
            total_tokens = 0
            total_oov = 0.0
            total_contractions = 0
            total_roman = 0
            start = time.perf_counter()
            for text in sampled:
                if method == "nltk":
                    tokens = self.tokenize_nltk(text)
                elif method == "spacy":
                    tokens = self.tokenize_spacy(text)
                else:
                    tokens = self.tokenize_custom(text)
                total_tokens += len(tokens)
                total_oov += self._oov_rate(tokens)
                total_contractions += self._contractions(tokens)
                total_roman += sum(1 for token in tokens if token.lower() in TextCleaner.ROMAN_URDU_MAP)
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "method": method,
                    "avg_tokens_per_doc": total_tokens / len(sampled),
                    "avg_oov_rate": total_oov / len(sampled),
                    "avg_contractions_per_doc": total_contractions / len(sampled),
                    "avg_roman_terms_per_doc": total_roman / len(sampled),
                    "seconds_total": elapsed,
                }
            )
        return pd.DataFrame(results)


class StopwordManager:
    def __init__(self):
        self.default = set(stopwords.words("english"))
        self.custom = self.build_custom_stopwords()

    def build_custom_stopwords(self) -> set[str]:
        preserve = {"not", "no", "never", "nor", "cannot", "don", "doesn", "didn", "won", "wouldn", "shouldn"}
        additions = {
            "also",
            "still",
            "even",
            "many",
            "much",
            "one",
            "two",
            "first",
            "second",
            "new",
            "according",
            "reported",
            "study",
            "says",
            "people",
            "case",
            "cases",
            "health",
            "vaccine",
            "virus",
            "covid",
        }
        custom = set(self.default) - preserve
        custom.update(additions)
        return custom

    @staticmethod
    def remove_stopwords(tokens: Sequence[str], stopword_set: set[str]) -> List[str]:
        return [token for token in tokens if token.lower() not in stopword_set]

    @staticmethod
    def removal_rate(tokens: Sequence[str], filtered: Sequence[str]) -> float:
        if len(tokens) == 0:
            return 0.0
        return 1.0 - (len(filtered) / len(tokens))


class StemLemmatizerComparer:
    def __init__(self):
        self.porter = PorterStemmer()
        self.snowball = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()

    def stem_porter(self, tokens: Sequence[str]) -> List[str]:
        return [self.porter.stem(token) for token in tokens]

    def stem_snowball(self, tokens: Sequence[str]) -> List[str]:
        return [self.snowball.stem(token) for token in tokens]

    def lemmatize(self, tokens: Sequence[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def compare(self, tokenized_texts: Sequence[Sequence[str]]) -> pd.DataFrame:
        methods = ["porter", "snowball", "lemmatizer"]
        rows = []
        for method in methods:
            start = time.perf_counter()
            transformed = []
            for tokens in tokenized_texts:
                if method == "porter":
                    transformed.append(self.stem_porter(tokens))
                elif method == "snowball":
                    transformed.append(self.stem_snowball(tokens))
                else:
                    transformed.append(self.lemmatize(tokens))
            elapsed = time.perf_counter() - start
            vocab_size = len({token for doc in transformed for token in doc})
            rows.append(
                {
                    "method": method,
                    "vocabulary_size": vocab_size,
                    "processing_seconds": elapsed,
                }
            )
        return pd.DataFrame(rows)

    def sample_domain_terms(self, terms: Sequence[str]) -> Dict[str, Dict[str, str]]:
        return {
            term: {
                "porter": self.porter.stem(term),
                "snowball": self.snowball.stem(term),
                "lemmatizer": self.lemmatizer.lemmatize(term),
            }
            for term in terms
        }


@dataclass
class FeatureEvaluation:
    accuracy: float
    precision: float
    recall: float
    f1: float


class FeatureBuilder:
    @staticmethod
    def build_bow(corpus: Sequence[str], max_features: Optional[int] = 20000) -> Tuple[CountVectorizer, np.ndarray]:
        vectorizer = CountVectorizer(max_features=max_features, token_pattern=r"\b[\w']+\b")
        matrix = vectorizer.fit_transform(corpus)
        return vectorizer, matrix

    @staticmethod
    def bow_sparsity(matrix: np.ndarray) -> float:
        if matrix.size == 0:
            return 0.0
        nonzeros = matrix.nnz
        return 1.0 - nonzeros / (matrix.shape[0] * matrix.shape[1])

    @staticmethod
    def build_tfidf(
        corpus: Sequence[str],
        smooth_idf: bool = False,
        sublinear_tf: bool = False,
        max_features: Optional[int] = 20000,
    ) -> Tuple[TfidfVectorizer, np.ndarray]:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r"\b[\w']+\b",
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )
        matrix = vectorizer.fit_transform(corpus)
        return vectorizer, matrix

    @staticmethod
    def top_terms_per_class(
        vectorizer: CountVectorizer,
        matrix: np.ndarray,
        labels: Sequence[str],
        top_n: int = 30,
    ) -> Dict[str, List[Tuple[str, float]]]:
        features = np.array(vectorizer.get_feature_names_out())
        class_scores = {}
        for label in sorted(set(labels)):
            mask = np.array([1 if lab == label else 0 for lab in labels], dtype=bool)
            class_matrix = matrix[mask]
            term_sums = np.asarray(class_matrix.sum(axis=0)).ravel()
            top_indices = term_sums.argsort()[::-1][:top_n]
            class_scores[label] = [(features[i], float(term_sums[i])) for i in top_indices]
        return class_scores

    @staticmethod
    def fit_word2vec(
        tokenized_texts: Sequence[Sequence[str]],
        sg: int = 0,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 3,
    ) -> Word2Vec:
        return Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=1,
            epochs=10,
        )

    @staticmethod
    def document_vectors(model: Word2Vec, tokenized_texts: Sequence[Sequence[str]]) -> np.ndarray:
        vectors = []
        for tokens in tokenized_texts:
            vectors.append(
                np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0)
                if any(token in model.wv for token in tokens)
                else np.zeros(model.vector_size, dtype=float)
            )
        return np.vstack(vectors)

    @staticmethod
    def cosine_retrieval(
        query: str,
        corpus: Sequence[str],
        vectorizer: TfidfVectorizer,
        matrix: np.ndarray,
        top_n: int = 10,
    ) -> List[Tuple[int, float]]:
        query_vec = vectorizer.transform([query])
        similarities = (matrix @ query_vec.T).toarray().ravel()
        norm = np.linalg.norm(query_vec.toarray())
        if norm == 0.0:
            return []
        doc_norms = np.linalg.norm(matrix.toarray(), axis=1)
        similarity_scores = similarities / np.maximum(doc_norms * norm, 1e-9)
        ranking = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
        return ranking[:top_n]

    @staticmethod
    def evaluate_classification(features: np.ndarray, labels: Sequence[str]) -> FeatureEvaluation:
        X_train, X_test, y_train, y_test = train_test_split(labels_to_matrix(features), labels, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return FeatureEvaluation(
            accuracy=accuracy_score(y_test, predictions),
            precision=precision_score(y_test, predictions, average="weighted", zero_division=0),
            recall=recall_score(y_test, predictions, average="weighted", zero_division=0),
            f1=f1_score(y_test, predictions, average="weighted", zero_division=0),
        )


def labels_to_matrix(features: np.ndarray) -> np.ndarray:
    return features


class NaiveBayesClassifier:
    def __init__(self):
        self.vectorizer: Optional[CountVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None

    def build(self, corpus: Sequence[str], labels: Sequence[str]) -> None:
        self.vectorizer = CountVectorizer(token_pattern=r"\b[\w']+\b", max_features=20000)
        matrix = self.vectorizer.fit_transform(corpus)
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(matrix, labels)

    def evaluate(self, corpus: Sequence[str], labels: Sequence[str]) -> FeatureEvaluation:
        assert self.vectorizer is not None and self.classifier is not None
        matrix = self.vectorizer.transform(corpus)
        preds = self.classifier.predict(matrix)
        return FeatureEvaluation(
            accuracy=accuracy_score(labels, preds),
            precision=precision_score(labels, preds, average="weighted", zero_division=0),
            recall=recall_score(labels, preds, average="weighted", zero_division=0),
            f1=f1_score(labels, preds, average="weighted", zero_division=0),
        )


class NGramLanguageModel:
    def __init__(self, n: int = 3, discount: float = 0.75):
        self.n = n
        self.discount = discount
        self.counts: Dict[Tuple[str, ...], int] = Counter()
        self.context_counts: Dict[Tuple[str, ...], int] = Counter()
        self.continuation_counts: Dict[Tuple[str, ...], int] = Counter()
        self.unique_contexts: Dict[str, set[str]] = defaultdict(set)
        self.vocabulary: set[str] = set()

    @staticmethod
    def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
        padded = ["<s>"] * (n - 1) + list(tokens) + ["</s>"]
        return [tuple(padded[i : i + n]) for i in range(len(padded) - n + 1)]

    def fit(self, tokenized_texts: Sequence[Sequence[str]]) -> None:
        for tokens in tokenized_texts:
            self.vocabulary.update(tokens)
            for ngram in self._ngrams(tokens, self.n):
                self.counts[ngram] += 1
                self.context_counts[ngram[:-1]] += 1
                self.unique_contexts[ngram[-1]].add(ngram[:-1])
        if self.n == 3:
            self._build_continuation_counts(tokenized_texts)

    def _build_continuation_counts(self, tokenized_texts: Sequence[Sequence[str]]) -> None:
        bigram_types = set()
        for tokens in tokenized_texts:
            for bigram in self._ngrams(tokens, 2):
                bigram_types.add(tuple(bigram))
        for bigram in bigram_types:
            self.continuation_counts[bigram] += 1

    def prob(self, context: Tuple[str, ...], token: str) -> float:
        if self.n == 1:
            return (self.counts[(token,)] + 1) / (sum(self.counts.values()) + len(self.vocabulary))
        if self.n == 2:
            bigram = context + (token,)
            numerator = max(self.counts[bigram] - self.discount, 0)
            denominator = self.context_counts[context] if self.context_counts[context] > 0 else 1
            lambda_weight = self.discount * len(self.unique_contexts[token]) / denominator
            continuation = (len(self.unique_contexts[token]) / len(self.counts)) if len(self.counts) else 0.0
            return numerator / denominator + lambda_weight * continuation
        if self.n == 3:
            trigram = context + (token,)
            trigram_count = self.counts[trigram]
            bigram_count = self.context_counts[context]
            numerator = max(trigram_count - self.discount, 0)
            lambda_weight = (
                self.discount * len(self.unique_contexts[token]) / bigram_count
                if bigram_count > 0
                else 0.0
            )
            continuation_prob = self.prob(context[1:], token) if context[1:] else 1.0 / max(len(self.vocabulary), 1)
            return numerator / bigram_count + lambda_weight * continuation_prob if bigram_count > 0 else continuation_prob
        return 0.0

    def perplexity(self, tokens: Sequence[str]) -> float:
        if len(tokens) == 0:
            return float("inf")
        ngrams = self._ngrams(tokens, self.n)
        log_prob = 0.0
        for gram in ngrams:
            context, token = gram[:-1], gram[-1]
            prob = self.prob(context, token)
            log_prob += math.log(prob + 1e-12)
        return math.exp(-log_prob / len(ngrams))

    def classify(self, tokenized_texts: Sequence[Sequence[str]], labels: Sequence[str]) -> Dict[str, float]:
        if self.n != 3:
            raise ValueError("Classification using Kneser-Ney is implemented only for trigram models")
        unique_labels = sorted(set(labels))
        if len(unique_labels) != 2:
            raise ValueError("Binary classification expects exactly two labels")
        holdout = list(zip(tokenized_texts, labels))
        random.shuffle(holdout)
        split = int(len(holdout) * 0.8)
        train = holdout[:split]
        test = holdout[split:]
        models = {}
        for label in unique_labels:
            texts = [tokens for tokens, lbl in train if lbl == label]
            model = NGramLanguageModel(n=3, discount=self.discount)
            model.fit(texts)
            models[label] = model
        predictions = []
        truths = []
        for tokens, label in test:
            scores = {lbl: model.perplexity(tokens) for lbl, model in models.items()}
            predicted = min(scores, key=scores.get)
            predictions.append(predicted)
            truths.append(label)
        return {
            "accuracy": accuracy_score(truths, predictions),
            "precision": precision_score(truths, predictions, average="weighted", zero_division=0),
            "recall": recall_score(truths, predictions, average="weighted", zero_division=0),
            "f1": f1_score(truths, predictions, average="weighted", zero_division=0),
        }


def build_text_pipeline_dataset(root_dir: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    loader = DatasetLoader()
    df = loader.load_covid19_fakenews(root_dir, sample_size)
    df["clean_text"] = df["text"].apply(TextCleaner.clean_text)
    return df


if __name__ == "__main__":
    data_dir = Path("datasets/covid19-fakenews")
    df = build_text_pipeline_dataset(data_dir, sample_size=500)
    print("Loaded dataset with shape", df.shape)
    audit = TextCleaner.audit_noise(df["text"], sample_size=200)
    print(audit.describe())
