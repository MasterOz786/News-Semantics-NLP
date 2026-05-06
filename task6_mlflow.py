from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from pandas.plotting import parallel_coordinates
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from nlp_pipeline import (
    DatasetLoader,
    StemLemmatizerComparer,
    StopwordManager,
    TextCleaner,
    TokenizerComparer,
)

EXPERIMENT_PREPROCESSING = "Assignment2/Preprocessing Ablation"
EXPERIMENT_FEATURES = "Assignment2/Feature Comparison"
EXPERIMENT_MODELS = "Assignment2/Model Comparison"
EXPERIMENT_BOOTSTRAP = "Assignment2/Serving Bootstrap"

REGISTRY_NAMES = {
    "naive_bayes": "FakeNewsNaiveBayes",
    "logistic_regression": "FakeNewsLogisticRegression",
    "polynomial_lr": "FakeNewsPolynomialLR",
}

SERVING_MODEL_NAME = REGISTRY_NAMES["logistic_regression"]
DEFAULT_SAMPLE_SIZE = 1500
RANDOM_STATE = 42


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


@dataclass
class PreprocessingConfig:
    name: str
    tokenizer: str = "custom"
    stopword_list: str = "none"
    normalization: str = "none"
    min_token_length: int = 1
    max_features: int = 5000
    vectorizer_type: str = "tfidf"
    sublinear_tf: bool = False
    smooth_idf: bool = True
    model_family: str = "logistic_regression"
    model_variant: str = "l2"
    degree: int = 2


@dataclass
class RunOutcome:
    family: str
    variant: str
    experiment_name: str
    run_id: str
    weighted_f1: float
    accuracy: float
    roc_auc: float
    training_time_seconds: float
    registered_name: Optional[str] = None
    registered_version: Optional[str] = None


class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def configure_mlflow(root_dir: Path) -> str:
    root = Path(root_dir).resolve()
    tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return tracking_uri


def get_client(root_dir: Path) -> MlflowClient:
    configure_mlflow(root_dir)
    return MlflowClient()


def _safe_experiment_dir(root_dir: Path, experiment_name: str) -> Path:
    slug = experiment_name.lower().replace("/", "_").replace(" ", "_")
    path = Path(root_dir).resolve() / "mlartifacts" / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_experiment(root_dir: Path, experiment_name: str) -> str:
    client = get_client(root_dir)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    artifact_location = _safe_experiment_dir(root_dir, experiment_name).as_uri()
    return client.create_experiment(
        experiment_name, artifact_location=artifact_location
    )


def load_base_dataset(
    root_dir: Path, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> pd.DataFrame:
    df = DatasetLoader.load_all_datasets(Path(root_dir), sample_size=sample_size)
    if df.empty:
        raise ValueError("No dataset rows available for MLflow experiments")
    df = df[["text", "label", "source"]].copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin({"fake", "real"})].reset_index(drop=True)
    return df


def get_split_indices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"].tolist(),
    )
    return train_idx, test_idx


def _tokenize(
    cleaned_text: str, tokenizer: TokenizerComparer, tokenizer_name: str
) -> List[str]:
    if tokenizer_name == "nltk":
        return tokenizer.tokenize_nltk(cleaned_text)
    if tokenizer_name == "spacy":
        return tokenizer.tokenize_spacy(cleaned_text)
    return tokenizer.tokenize_custom(cleaned_text)


def _normalize_tokens(
    tokens: Sequence[str], normalization: str, stemmer: StemLemmatizerComparer
) -> List[str]:
    if normalization == "porter":
        return stemmer.stem_porter(tokens)
    if normalization == "snowball":
        return stemmer.stem_snowball(tokens)
    if normalization == "lemmatizer":
        return stemmer.lemmatize(tokens)
    return list(tokens)


def process_text_for_config(
    text: str,
    config: PreprocessingConfig,
    tokenizer: TokenizerComparer,
    stopword_manager: StopwordManager,
    stemmer: StemLemmatizerComparer,
) -> Tuple[str, List[str], List[str]]:
    cleaned = TextCleaner.clean_text(text)
    tokens = _tokenize(cleaned, tokenizer, config.tokenizer)

    removed_stopwords: List[str] = []
    if config.stopword_list == "default":
        removed_stopwords = [
            token for token in tokens if token.lower() in stopword_manager.default
        ]
        tokens = stopword_manager.remove_stopwords(tokens, stopword_manager.default)
    elif config.stopword_list == "custom":
        removed_stopwords = [
            token for token in tokens if token.lower() in stopword_manager.custom
        ]
        tokens = stopword_manager.remove_stopwords(tokens, stopword_manager.custom)

    tokens = _normalize_tokens(tokens, config.normalization, stemmer)
    tokens = [
        token
        for token in tokens
        if len(token) >= config.min_token_length and token.strip()
    ]

    if not tokens:
        fallback = cleaned.strip() if cleaned.strip() else "empty"
        tokens = [fallback]

    return " ".join(tokens), tokens, removed_stopwords


def preprocess_corpus(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    tokenizer = TokenizerComparer()
    stopword_manager = StopwordManager()
    stemmer = StemLemmatizerComparer()

    rows = []
    for _, row in df.iterrows():
        processed_text, tokens, removed = process_text_for_config(
            row["text"], config, tokenizer, stopword_manager, stemmer
        )
        rows.append(
            {
                "processed_text": processed_text,
                "tokens": tokens,
                "removed_stopwords": removed,
                "label": row["label"],
                "source": row["source"],
            }
        )
    return pd.DataFrame(rows)


def build_pipeline(config: PreprocessingConfig) -> Pipeline:
    if config.vectorizer_type == "bow":
        vectorizer = CountVectorizer(
            max_features=config.max_features,
            token_pattern=r"\b[\w']+\b",
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            token_pattern=r"\b[\w']+\b",
            smooth_idf=config.smooth_idf,
            sublinear_tf=config.sublinear_tf,
        )

    if config.model_family == "naive_bayes":
        classifier = MultinomialNB(
            alpha=0.1 if config.model_variant == "alpha_0.1" else 1.0
        )
        return Pipeline(
            [
                ("vectorizer", vectorizer),
                ("classifier", classifier),
            ]
        )

    if config.model_family == "logistic_regression":
        lr_kwargs: Dict[str, Any] = {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
        }
        if config.model_variant == "l1":
            lr_kwargs.update({"penalty": "l1", "solver": "saga"})
        elif config.model_variant == "elasticnet":
            lr_kwargs.update(
                {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5}
            )
        else:
            lr_kwargs.update({"penalty": "l2", "solver": "lbfgs"})
        classifier = LogisticRegression(**lr_kwargs)
        return Pipeline(
            [
                ("vectorizer", vectorizer),
                ("classifier", classifier),
            ]
        )

    if config.model_family == "polynomial_lr":
        classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        return Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        max_features=config.max_features, token_pattern=r"\b[\w']+\b"
                    ),
                ),
                ("to_dense", DenseTransformer()),
                ("pca", PCA(n_components=2, random_state=RANDOM_STATE)),
                ("poly", PolynomialFeatures(degree=config.degree, include_bias=False)),
                ("classifier", classifier),
            ]
        )

    raise ValueError(f"Unsupported model family: {config.model_family}")


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: Sequence[str],
    y_train: Sequence[str],
    X_test: Sequence[str],
    y_test: Sequence[str],
) -> Dict[str, Any]:
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    predictions = pipeline.predict(X_test)
    probabilities = (
        pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    )

    weighted_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_test, predictions)
    per_class = classification_report(
        y_test, predictions, zero_division=0, output_dict=True
    )

    classes = list(getattr(pipeline, "classes_", [])) or list(
        getattr(pipeline.named_steps["classifier"], "classes_", [])
    )
    roc_auc = 0.0
    if probabilities is not None and len(classes) == 2:
        positive_class = sorted(classes)[1]
        positive_index = classes.index(positive_class)
        y_true = np.array([1 if label == positive_class else 0 for label in y_test])
        fpr, tpr, _ = roc_curve(y_true, probabilities[:, positive_index])
        roc_auc = float(auc(fpr, tpr))
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    return {
        "pipeline": pipeline,
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "training_time_seconds": float(training_time),
        "roc_auc": float(roc_auc),
        "fpr": fpr,
        "tpr": tpr,
    }


def write_artifacts(
    output_dir: Path,
    pipeline: Pipeline,
    y_test: Sequence[str],
    predictions: Sequence[str],
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    per_class: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, predictions, labels=sorted(set(y_test)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=sorted(set(y_test))
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    vectorizer = pipeline.named_steps.get("vectorizer")
    vocabulary = (
        vectorizer.get_feature_names_out().tolist() if vectorizer is not None else []
    )
    (output_dir / "tfidf_vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2), encoding="utf-8"
    )

    (output_dir / "classification_report.json").write_text(
        json.dumps(per_class, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0), encoding="utf-8"
    )


def log_run_metadata(
    config: PreprocessingConfig,
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    accuracy: float,
    weighted_f1: float,
    roc_auc: float,
    training_time_seconds: float,
    per_class: Dict[str, Any],
) -> None:
    mlflow.log_param(
        "dataset_sources",
        json.dumps(sorted(df["source"].astype(str).unique().tolist())),
    )
    mlflow.log_param("train_size", train_size)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("tokenizer", config.tokenizer)
    mlflow.log_param("stopword_list", config.stopword_list)
    mlflow.log_param("normalization_method", config.normalization)
    mlflow.log_param("min_token_length", config.min_token_length)
    mlflow.log_param("vectorizer_type", config.vectorizer_type)
    mlflow.log_param(
        "vectorizer_settings",
        json.dumps(
            {
                "max_features": config.max_features,
                "sublinear_tf": config.sublinear_tf,
                "smooth_idf": config.smooth_idf,
            }
        ),
    )
    mlflow.log_param("model_type", config.model_family)
    mlflow.log_param("model_variant", config.model_variant)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("weighted_f1", weighted_f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("training_time_seconds", training_time_seconds)

    for label in [label for label in per_class.keys() if label in {"fake", "real"}]:
        mlflow.log_metric(f"precision_{label}", float(per_class[label]["precision"]))
        mlflow.log_metric(f"recall_{label}", float(per_class[label]["recall"]))
        mlflow.log_metric(f"f1_{label}", float(per_class[label]["f1-score"]))


def run_single_experiment(
    root_dir: Path,
    experiment_name: str,
    parent_run_name: str,
    config: PreprocessingConfig,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    nested: bool = True,
    register_name: Optional[str] = None,
) -> RunOutcome:
    experiment_id = ensure_experiment(root_dir, experiment_name)
    processed = preprocess_corpus(df, config)

    X_train = processed.iloc[train_idx]["processed_text"].tolist()
    y_train = processed.iloc[train_idx]["label"].tolist()
    X_test = processed.iloc[test_idx]["processed_text"].tolist()
    y_test = processed.iloc[test_idx]["label"].tolist()

    pipeline = build_pipeline(config)
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=config.name, nested=nested
    ) as run:
        mlflow.set_tag("run_group", parent_run_name)
        mlflow.set_tag("config_name", config.name)
        result = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)
        log_run_metadata(
            config,
            df,
            train_size=len(X_train),
            test_size=len(X_test),
            accuracy=result["accuracy"],
            weighted_f1=result["weighted_f1"],
            roc_auc=result["roc_auc"],
            training_time_seconds=result["training_time_seconds"],
            per_class=result["per_class"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            write_artifacts(
                tmp_dir,
                result["pipeline"],
                y_test,
                result["predictions"],
                result["fpr"],
                result["tpr"],
                result["roc_auc"],
                result["per_class"],
            )
            mlflow.log_artifacts(str(tmp_dir), artifact_path="artifacts")

        mlflow.sklearn.log_model(result["pipeline"], artifact_path="model")

        registered_version = None
        if register_name:
            model_uri = f"runs:/{run.info.run_id}/model"
            version = mlflow.register_model(model_uri=model_uri, name=register_name)
            registered_version = str(version.version)
            client = get_client(root_dir)
            client.set_model_version_tag(
                register_name,
                version.version,
                "weighted_f1",
                f"{result['weighted_f1']:.6f}",
            )
            client.set_model_version_tag(
                register_name, version.version, "accuracy", f"{result['accuracy']:.6f}"
            )
            client.set_model_version_tag(
                register_name, version.version, "roc_auc", f"{result['roc_auc']:.6f}"
            )
            client.set_model_version_tag(
                register_name, version.version, "model_family", config.model_family
            )
            client.transition_model_version_stage(
                register_name, version.version, stage="Staging"
            )
            maybe_promote_model(
                root_dir, register_name, str(version.version), result["weighted_f1"]
            )

        return RunOutcome(
            family=config.model_family,
            variant=config.model_variant,
            experiment_name=experiment_name,
            run_id=run.info.run_id,
            weighted_f1=result["weighted_f1"],
            accuracy=result["accuracy"],
            roc_auc=result["roc_auc"],
            training_time_seconds=result["training_time_seconds"],
            registered_name=register_name,
            registered_version=registered_version,
        )


def maybe_promote_model(
    root_dir: Path, model_name: str, candidate_version: str, candidate_f1: float
) -> None:
    client = get_client(root_dir)
    latest_prod = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_prod:
        client.transition_model_version_stage(
            model_name,
            candidate_version,
            stage="Production",
            archive_existing_versions=True,
        )
        return

    current_prod = latest_prod[0]
    current_prod_f1 = float(current_prod.tags.get("weighted_f1", 0.0))
    if candidate_f1 >= current_prod_f1 + 0.01:
        client.transition_model_version_stage(
            model_name,
            candidate_version,
            stage="Production",
            archive_existing_versions=True,
        )


def _log_parallel_coordinates_artifact(
    results: List[Dict[str, Any]], output_dir: Path
) -> Path:
    plot_df = pd.DataFrame(results)
    plot_df["configuration"] = plot_df["name"]
    plot_df["stopword_code"] = plot_df["stopword_list"].map(
        {"none": 0, "default": 1, "custom": 2}
    )
    plot_df["norm_code"] = plot_df["normalization"].map(
        {"none": 0, "porter": 1, "snowball": 2, "lemmatizer": 3}
    )
    plot_df["tokenizer_code"] = plot_df["tokenizer"].map(
        {"custom": 0, "nltk": 1, "spacy": 2}
    )

    cols = [
        "configuration",
        "weighted_f1",
        "accuracy",
        "roc_auc",
        "min_token_length",
        "max_features",
        "stopword_code",
        "norm_code",
        "tokenizer_code",
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    parallel_coordinates(
        plot_df[cols], class_column="configuration", ax=ax, colormap="viridis"
    )
    ax.set_title("Preprocessing Ablation Parallel Coordinates")
    ax.set_ylabel("F1-weighted and feature settings")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "parallel_coordinates_preprocessing_ablation.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_preprocessing_ablation(
    root_dir: Path, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> List[RunOutcome]:
    df = load_base_dataset(root_dir, sample_size=sample_size)
    train_idx, test_idx = get_split_indices(df)
    experiment_id = ensure_experiment(root_dir, EXPERIMENT_PREPROCESSING)

    configs = [
        PreprocessingConfig(
            name="baseline_no_stopwords",
            stopword_list="none",
            normalization="none",
            min_token_length=1,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="default_stopwords",
            stopword_list="default",
            normalization="none",
            min_token_length=1,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="custom_stopwords",
            stopword_list="custom",
            normalization="none",
            min_token_length=1,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="porter_default_stopwords",
            stopword_list="default",
            normalization="porter",
            min_token_length=2,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="lemmatizer_custom_stopwords",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="lemmatizer_custom_high_features",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=3,
            max_features=10000,
        ),
    ]

    outcomes: List[RunOutcome] = []
    plot_rows: List[Dict[str, Any]] = []
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="preprocessing_ablation_group"
    ) as parent_run:
        mlflow.set_tag("run_group", "preprocessing_ablation")
        for config in configs:
            outcome = run_single_experiment(
                root_dir=root_dir,
                experiment_name=EXPERIMENT_PREPROCESSING,
                parent_run_name="preprocessing_ablation_group",
                config=config,
                df=df,
                train_idx=train_idx,
                test_idx=test_idx,
                nested=True,
            )
            outcomes.append(outcome)
            plot_rows.append(
                {
                    "name": config.name,
                    "tokenizer": config.tokenizer,
                    "stopword_list": config.stopword_list,
                    "normalization": config.normalization,
                    "min_token_length": config.min_token_length,
                    "max_features": config.max_features,
                    "weighted_f1": outcome.weighted_f1,
                    "accuracy": outcome.accuracy,
                    "roc_auc": outcome.roc_auc,
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            plot_path = _log_parallel_coordinates_artifact(plot_rows, tmp_dir)
            mlflow.log_artifact(str(plot_path), artifact_path="summary")
            (tmp_dir / "preprocessing_ablation_results.json").write_text(
                json.dumps(plot_rows, indent=2), encoding="utf-8"
            )
            mlflow.log_artifact(
                str(tmp_dir / "preprocessing_ablation_results.json"),
                artifact_path="summary",
            )
    return outcomes


def run_feature_comparison(
    root_dir: Path, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> List[RunOutcome]:
    df = load_base_dataset(root_dir, sample_size=sample_size)
    train_idx, test_idx = get_split_indices(df)
    experiment_id = ensure_experiment(root_dir, EXPERIMENT_FEATURES)

    configs = [
        PreprocessingConfig(
            name="bow_lr",
            vectorizer_type="bow",
            model_family="logistic_regression",
            model_variant="l2",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="tfidf_lr",
            vectorizer_type="tfidf",
            model_family="logistic_regression",
            model_variant="l2",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="tfidf_sublinear_lr",
            vectorizer_type="tfidf",
            sublinear_tf=True,
            model_family="logistic_regression",
            model_variant="l2",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        ),
    ]

    outcomes: List[RunOutcome] = []
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="feature_comparison_group"
    ) as parent_run:
        mlflow.set_tag("run_group", "feature_comparison")
        for config in configs:
            outcomes.append(
                run_single_experiment(
                    root_dir=root_dir,
                    experiment_name=EXPERIMENT_FEATURES,
                    parent_run_name="feature_comparison_group",
                    config=config,
                    df=df,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    nested=True,
                )
            )
    return outcomes


def run_model_comparison(
    root_dir: Path, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> List[RunOutcome]:
    df = load_base_dataset(root_dir, sample_size=sample_size)
    train_idx, test_idx = get_split_indices(df)
    experiment_id = ensure_experiment(root_dir, EXPERIMENT_MODELS)

    configs = [
        PreprocessingConfig(
            name="nb_alpha_0_1",
            model_family="naive_bayes",
            model_variant="alpha_0.1",
            vectorizer_type="bow",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="nb_alpha_1_0",
            model_family="naive_bayes",
            model_variant="alpha_1.0",
            vectorizer_type="bow",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="lr_l1",
            model_family="logistic_regression",
            model_variant="l1",
            vectorizer_type="tfidf",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="lr_l2",
            model_family="logistic_regression",
            model_variant="l2",
            vectorizer_type="tfidf",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="lr_elasticnet",
            model_family="logistic_regression",
            model_variant="elasticnet",
            vectorizer_type="tfidf",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        ),
        PreprocessingConfig(
            name="poly_degree_1",
            model_family="polynomial_lr",
            model_variant="degree_1",
            vectorizer_type="tfidf",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
            degree=1,
        ),
        PreprocessingConfig(
            name="poly_degree_2",
            model_family="polynomial_lr",
            model_variant="degree_2",
            vectorizer_type="tfidf",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
            degree=2,
        ),
        PreprocessingConfig(
            name="poly_degree_3",
            model_family="polynomial_lr",
            model_variant="degree_3",
            vectorizer_type="tfidf",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
            degree=3,
        ),
    ]

    outcomes: List[RunOutcome] = []
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="model_comparison_group"
    ) as parent_run:
        mlflow.set_tag("run_group", "model_comparison")
        raw_outcomes: List[Tuple[PreprocessingConfig, RunOutcome]] = []
        for config in configs:
            outcome = run_single_experiment(
                root_dir=root_dir,
                experiment_name=EXPERIMENT_MODELS,
                parent_run_name="model_comparison_group",
                config=config,
                df=df,
                train_idx=train_idx,
                test_idx=test_idx,
                nested=True,
            )
            raw_outcomes.append((config, outcome))
            outcomes.append(outcome)

    best_by_family: Dict[str, Tuple[PreprocessingConfig, RunOutcome]] = {}
    for config, outcome in raw_outcomes:
        current = best_by_family.get(config.model_family)
        if current is None or outcome.weighted_f1 > current[1].weighted_f1:
            best_by_family[config.model_family] = (config, outcome)

    registered_outcomes: List[RunOutcome] = []
    for family, (config, _) in best_by_family.items():
        register_name = REGISTRY_NAMES[family]
        registered_outcomes.append(
            run_single_experiment(
                root_dir=root_dir,
                experiment_name=EXPERIMENT_MODELS,
                parent_run_name="model_registry_registration",
                config=config,
                df=df,
                train_idx=train_idx,
                test_idx=test_idx,
                nested=False,
                register_name=register_name,
            )
        )

    return outcomes + registered_outcomes


def run_all_task6(
    root_dir: Path = Path("."), sample_size: int = DEFAULT_SAMPLE_SIZE
) -> Dict[str, List[Dict[str, Any]]]:
    configure_mlflow(root_dir)
    preprocessing = run_preprocessing_ablation(root_dir, sample_size=sample_size)
    features = run_feature_comparison(root_dir, sample_size=sample_size)
    models = run_model_comparison(root_dir, sample_size=sample_size)
    return {
        "preprocessing_ablation": [asdict(item) for item in preprocessing],
        "feature_comparison": [asdict(item) for item in features],
        "model_comparison": [asdict(item) for item in models],
    }


def ensure_serving_logistic_model(
    root_dir: Path = Path("."), sample_size: int = DEFAULT_SAMPLE_SIZE
) -> Dict[str, Any]:
    client = get_client(root_dir)
    try:
        versions = client.search_model_versions(f"name = '{SERVING_MODEL_NAME}'")
    except Exception:
        versions = []

    if not versions:
        experiment_id = ensure_experiment(root_dir, EXPERIMENT_BOOTSTRAP)
        df = load_base_dataset(root_dir, sample_size=sample_size)
        train_idx, test_idx = get_split_indices(df)
        config = PreprocessingConfig(
            name="bootstrap_lr_l2",
            model_family="logistic_regression",
            model_variant="l2",
            vectorizer_type="tfidf",
            stopword_list="custom",
            normalization="lemmatizer",
            min_token_length=2,
            max_features=5000,
        )
        run_single_experiment(
            root_dir=root_dir,
            experiment_name=EXPERIMENT_BOOTSTRAP,
            parent_run_name="bootstrap_serving_model",
            config=config,
            df=df,
            train_idx=train_idx,
            test_idx=test_idx,
            nested=False,
            register_name=SERVING_MODEL_NAME,
        )
        versions = client.search_model_versions(f"name = '{SERVING_MODEL_NAME}'")

    production = client.get_latest_versions(SERVING_MODEL_NAME, stages=["Production"])
    target = (
        production[0]
        if production
        else sorted(versions, key=lambda item: int(item.version))[-1]
    )
    run = client.get_run(target.run_id)
    return {
        "model_name": SERVING_MODEL_NAME,
        "model_version": str(target.version),
        "stage": target.current_stage or "None",
        "run_id": target.run_id,
        "weighted_f1": safe_float(
            target.tags.get("weighted_f1", run.data.metrics.get("weighted_f1", 0.0))
        ),
        "accuracy": safe_float(
            target.tags.get("accuracy", run.data.metrics.get("accuracy", 0.0))
        ),
        "roc_auc": safe_float(
            target.tags.get("roc_auc", run.data.metrics.get("roc_auc", 0.0))
        ),
        "model_uri": f"models:/{SERVING_MODEL_NAME}/{target.current_stage}"
        if target.current_stage
        else f"models:/{SERVING_MODEL_NAME}/{target.version}",
    }


def get_model_version_history(
    root_dir: Path, model_names: Optional[Iterable[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    client = get_client(root_dir)
    names = list(model_names or REGISTRY_NAMES.values())
    history: Dict[str, List[Dict[str, Any]]] = {}
    for model_name in names:
        try:
            versions = client.search_model_versions(f"name = '{model_name}'")
        except Exception:
            history[model_name] = []
            continue
        sorted_versions = sorted(versions, key=lambda item: int(item.version))
        entries: List[Dict[str, Any]] = []
        for version in sorted_versions:
            run = client.get_run(version.run_id)
            entries.append(
                {
                    "version": str(version.version),
                    "stage": version.current_stage or "None",
                    "run_id": version.run_id,
                    "weighted_f1": safe_float(
                        version.tags.get(
                            "weighted_f1", run.data.metrics.get("weighted_f1", 0.0)
                        )
                    ),
                    "accuracy": safe_float(
                        version.tags.get(
                            "accuracy", run.data.metrics.get("accuracy", 0.0)
                        )
                    ),
                    "roc_auc": safe_float(
                        version.tags.get(
                            "roc_auc", run.data.metrics.get("roc_auc", 0.0)
                        )
                    ),
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                }
            )
        history[model_name] = entries
    return history


def get_recent_run_summaries(
    root_dir: Path, max_results: int = 10
) -> List[Dict[str, Any]]:
    client = get_client(root_dir)
    experiment_ids = []
    for experiment_name in [
        EXPERIMENT_PREPROCESSING,
        EXPERIMENT_FEATURES,
        EXPERIMENT_MODELS,
        EXPERIMENT_BOOTSTRAP,
    ]:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_ids.append(experiment.experiment_id)
    if not experiment_ids:
        return []

    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        output_format="pandas",
        max_results=max_results,
        order_by=["attribute.start_time DESC"],
    )
    if runs.empty:
        return []

    summaries: List[Dict[str, Any]] = []
    for _, row in runs.iterrows():
        summaries.append(
            {
                "run_id": row.get("run_id"),
                "experiment_id": row.get("experiment_id"),
                "status": row.get("status"),
                "run_name": row.get("tags.mlflow.runName"),
                "weighted_f1": safe_float(row.get("metrics.weighted_f1", 0.0) or 0.0),
                "accuracy": safe_float(row.get("metrics.accuracy", 0.0) or 0.0),
                "roc_auc": safe_float(row.get("metrics.roc_auc", 0.0) or 0.0),
                "start_time": row.get("start_time").isoformat()
                if hasattr(row.get("start_time"), "isoformat")
                else str(row.get("start_time")),
            }
        )
    return summaries


if __name__ == "__main__":
    summary = run_all_task6(Path("."))
    print(json.dumps(summary, indent=2))
