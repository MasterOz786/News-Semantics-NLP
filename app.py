from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from nlp_pipeline import (
    DatasetLoader,
    StemLemmatizerComparer,
    StopwordManager,
    TextCleaner,
    TokenizerComparer,
)
from task6_mlflow import (
    PreprocessingConfig,
    configure_mlflow,
    ensure_serving_logistic_model,
    get_model_version_history,
    get_recent_run_summaries,
    process_text_for_config,
)

ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger("task7_api")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


class RateLimiter:
    def __init__(self):
        self.windows: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)

    def check(self, key: Tuple[str, str], limit: int, window_seconds: int = 60) -> bool:
        now = time.monotonic()
        queue = self.windows[key]
        while queue and now - queue[0] > window_seconds:
            queue.popleft()
        if len(queue) >= limit:
            return False
        queue.append(now)
        return True


@dataclass
class AppResources:
    model: Any
    model_info: Dict[str, Any]
    tokenizer: TokenizerComparer
    stopword_manager: StopwordManager
    stemmer: StemLemmatizerComparer
    retrieval_vectorizer: Any
    retrieval_matrix: Any
    retrieval_claims: List[Dict[str, Any]]
    load_timestamp: str
    rate_limiter: RateLimiter


class ProcessingSteps(BaseModel):
    tokenizer: Literal["custom", "nltk", "spacy"] = "custom"
    stopword_list: Literal["none", "default", "custom"] = "custom"
    normalization: Literal["none", "porter", "snowball", "lemmatizer"] = "lemmatizer"
    min_token_length: int = Field(default=2, ge=1, le=20)


class PreprocessRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)
    steps: ProcessingSteps = Field(default_factory=ProcessingSteps)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

    @validator("texts")
    def validate_texts(cls, texts: List[str]) -> List[str]:
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Each batch item must be a string")
            if len(text) < 10 or len(text) > 10000:
                raise ValueError("Each text must be between 10 and 10000 characters")
        return texts


class SimilarRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)


def get_client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def make_processing_config(steps: ProcessingSteps) -> PreprocessingConfig:
    return PreprocessingConfig(
        name="api_preprocess",
        tokenizer=steps.tokenizer,
        stopword_list=steps.stopword_list,
        normalization=steps.normalization,
        min_token_length=steps.min_token_length,
        max_features=5000,
        vectorizer_type="tfidf",
        model_family="logistic_regression",
        model_variant="l2",
    )


def preprocess_text(
    text: str, steps: ProcessingSteps, resources: AppResources
) -> Dict[str, Any]:
    config = make_processing_config(steps)
    start = time.perf_counter()
    processed_text, tokens, removed_stopwords = process_text_for_config(
        text=text,
        config=config,
        tokenizer=resources.tokenizer,
        stopword_manager=resources.stopword_manager,
        stemmer=resources.stemmer,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "cleaned_text": processed_text,
        "tokens": tokens,
        "removed_stopwords": removed_stopwords,
        "processing_time_ms": round(elapsed_ms, 3),
    }


def classify_text(text: str, resources: AppResources) -> Dict[str, Any]:
    processed = preprocess_text(text, ProcessingSteps(), resources)
    model = resources.model
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]

    X = vectorizer.transform([processed["cleaned_text"]])
    probabilities = classifier.predict_proba(X)[0]
    classes = classifier.classes_.tolist()
    predicted_index = int(np.argmax(probabilities))
    prediction = classes[predicted_index]
    confidence = float(probabilities[predicted_index])

    feature_names = vectorizer.get_feature_names_out()
    x_dense = X.toarray()[0]
    coef = classifier.coef_[0]
    if prediction == classes[1]:
        contribution_scores = x_dense * coef
    else:
        contribution_scores = x_dense * (-coef)

    nonzero_indices = np.where(x_dense > 0)[0]
    ranked = sorted(
        [(feature_names[i], float(contribution_scores[i])) for i in nonzero_indices],
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:10]

    return {
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "class_probabilities": {
            classes[i]: round(float(probabilities[i]), 6) for i in range(len(classes))
        },
        "top_contributing_features": [
            {"feature": feature, "contribution": round(score, 6)}
            for feature, score in ranked
        ],
        "preprocessed_text": processed["cleaned_text"],
        "processing_steps": {
            "tokenizer": "custom",
            "stopword_list": "custom",
            "normalization": "lemmatizer",
            "min_token_length": 2,
        },
    }


def classify_batch(texts: List[str], resources: AppResources) -> Dict[str, Any]:
    start = time.perf_counter()
    processed_texts = [
        preprocess_text(text, ProcessingSteps(), resources)["cleaned_text"]
        for text in texts
    ]
    model = resources.model
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    X = vectorizer.transform(processed_texts)
    probabilities = classifier.predict_proba(X)
    classes = classifier.classes_.tolist()
    predictions = classifier.predict(X)

    results = []
    for idx, prediction in enumerate(predictions):
        predicted_index = classes.index(prediction)
        results.append(
            {
                "prediction": prediction,
                "confidence": round(float(probabilities[idx][predicted_index]), 6),
                "class_probabilities": {
                    classes[i]: round(float(probabilities[idx][i]), 6)
                    for i in range(len(classes))
                },
            }
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "count": len(texts),
        "processing_time_ms": round(elapsed_ms, 3),
        "results": results,
    }


def retrieve_similar_claims(
    text: str, top_k: int, resources: AppResources
) -> Dict[str, Any]:
    cleaned_query = TextCleaner.clean_text(text)
    query_vec = resources.retrieval_vectorizer.transform([cleaned_query])
    similarities = (resources.retrieval_matrix @ query_vec.T).toarray().ravel()
    query_norm = np.linalg.norm(query_vec.toarray())
    if query_norm == 0.0:
        results: List[Dict[str, Any]] = []
    else:
        matrix_dense = resources.retrieval_matrix.toarray()
        doc_norms = np.linalg.norm(matrix_dense, axis=1)
        scores = similarities / np.maximum(doc_norms * query_norm, 1e-9)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            {
                "claim": resources.retrieval_claims[int(i)]["claim"],
                "label": resources.retrieval_claims[int(i)]["label"],
                "source": resources.retrieval_claims[int(i)]["source"],
                "cosine_similarity": round(float(scores[int(i)]), 6),
            }
            for i in top_indices
        ]
    return {
        "query": cleaned_query,
        "top_k": top_k,
        "results": results,
    }


def build_retrieval_bundle(
    resources: AppResources | None = None,
) -> Tuple[Any, Any, List[Dict[str, Any]]]:
    liar_df = DatasetLoader.load_liar_dataset(ROOT_DIR)
    liar_df = liar_df.dropna().reset_index(drop=True)
    claims = [
        {
            "claim": TextCleaner.clean_text(row["text"]),
            "label": str(row["label"]),
            "source": str(row["source"]),
        }
        for _, row in liar_df.iterrows()
        if str(row["text"]).strip()
    ]
    corpus = [item["claim"] for item in claims]
    vectorizer = (
        resources.model.named_steps["vectorizer"].__class__(
            **resources.model.named_steps["vectorizer"].get_params()
        )
        if resources is not None
        else None
    )
    if vectorizer is None:
        raise ValueError(
            "Resources must be initialized before retrieval bundle construction"
        )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, claims


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_mlflow(ROOT_DIR)
    model_info = ensure_serving_logistic_model(ROOT_DIR)
    model = mlflow.sklearn.load_model(model_info["model_uri"])
    tokenizer = TokenizerComparer()
    stopword_manager = StopwordManager()
    stemmer = StemLemmatizerComparer()
    base_resources = AppResources(
        model=model,
        model_info=model_info,
        tokenizer=tokenizer,
        stopword_manager=stopword_manager,
        stemmer=stemmer,
        retrieval_vectorizer=None,
        retrieval_matrix=None,
        retrieval_claims=[],
        load_timestamp=datetime.now(timezone.utc).isoformat(),
        rate_limiter=RateLimiter(),
    )
    retrieval_vectorizer, retrieval_matrix, retrieval_claims = build_retrieval_bundle(
        base_resources
    )
    base_resources.retrieval_vectorizer = retrieval_vectorizer
    base_resources.retrieval_matrix = retrieval_matrix
    base_resources.retrieval_claims = retrieval_claims
    app.state.resources = base_resources
    LOGGER.info(
        "Loaded model %s version=%s stage=%s weighted_f1=%.4f",
        model_info["model_name"],
        model_info["model_version"],
        model_info["stage"],
        model_info["weighted_f1"],
    )
    yield


app = FastAPI(title="Fake News Inference API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    LOGGER.info(
        "%s %s status=%s client=%s time_ms=%.3f",
        request.method,
        request.url.path,
        response.status_code,
        get_client_key(request),
        elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
async def health(request: Request):
    resources: AppResources = request.app.state.resources
    f1_value = round(float(resources.model_info["weighted_f1"]), 6)
    return {
        "model_name": resources.model_info["model_name"],
        "model_version": resources.model_info["model_version"],
        "stage": resources.model_info["stage"],
        "f1_score": f1_value,
        "weighted_f1": f1_value,
        "load_timestamp": resources.load_timestamp,
    }


@app.post("/preprocess")
async def preprocess(request: Request, payload: PreprocessRequest):
    resources: AppResources = request.app.state.resources
    result = preprocess_text(payload.text, payload.steps, resources)
    return result


@app.post("/classify")
async def classify(request: Request, payload: ClassifyRequest):
    resources: AppResources = request.app.state.resources
    key = (get_client_key(request), "/classify")
    if not resources.rate_limiter.check(key, limit=100):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for /classify")
    start = time.perf_counter()
    result = classify_text(payload.text, resources)
    result["processing_time_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return result


@app.post("/classify/batch")
async def classify_batch_endpoint(request: Request, payload: BatchClassifyRequest):
    resources: AppResources = request.app.state.resources
    key = (get_client_key(request), "/classify/batch")
    if not resources.rate_limiter.check(key, limit=10):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded for /classify/batch"
        )
    return classify_batch(payload.texts, resources)


@app.post("/retrieve/similar")
async def retrieve_similar(request: Request, payload: SimilarRequest):
    resources: AppResources = request.app.state.resources
    return retrieve_similar_claims(payload.text, payload.top_k, resources)


@app.get("/model/performance")
async def model_performance(request: Request):
    resources: AppResources = request.app.state.resources
    version_history = get_model_version_history(ROOT_DIR)
    recent_runs = get_recent_run_summaries(ROOT_DIR)
    return {
        "current_model": {
            "model_name": resources.model_info["model_name"],
            "model_version": resources.model_info["model_version"],
            "stage": resources.model_info["stage"],
            "f1_score": round(float(resources.model_info["weighted_f1"]), 6),
            "weighted_f1": round(float(resources.model_info["weighted_f1"]), 6),
            "accuracy": round(float(resources.model_info["accuracy"]), 6),
            "roc_auc": round(float(resources.model_info["roc_auc"]), 6),
        },
        "version_history": version_history,
        "recent_runs": recent_runs,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
