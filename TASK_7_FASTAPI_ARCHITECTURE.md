# Task 7 — FastAPI Inference System Architecture

## Purpose

This document defines the FastAPI inference architecture **before implementation** for Task 7. It covers:

- startup model loading via lifespan context manager
- request logging middleware
- rate limiting
- preprocessing, classification, batch inference, retrieval, and model performance endpoints
- MLflow integration for live metrics and version history

---

## Architecture Diagram

```mermaid
flowchart TD
    A[Client Request] --> B[FastAPI App]
    B --> C[Logging Middleware\nConsole + Rotating File]
    C --> D[Rate Limiter\n/classify 100 req/min\n/classify/batch 10 req/min]
    D --> E[Endpoint Router]

    subgraph Startup[Lifespan Startup]
        S1[Configure MLflow\ntracking URI + registry URI]
        S2[Load or Train Serving Bundle]
        S3[Load Retrieval Corpus\nFact-checked claims]
        S4[Initialize Shared Services\nCleaner / Tokenizer / Stopwords / Stemmer]
        S5[Store model + metadata in app.state]
        S1 --> S2 --> S3 --> S4 --> S5
    end

    E --> F1[GET /health]
    E --> F2[POST /preprocess]
    E --> F3[POST /classify]
    E --> F4[POST /classify/batch]
    E --> F5[POST /retrieve/similar]
    E --> F6[GET /model/performance]

    F2 --> G1[Preprocessing Service\nclean -> tokenize -> stopword removal -> normalization]
    F3 --> G2[Classification Service\nvectorize -> predict -> probabilities -> top features]
    F4 --> G3[Batch Classification Service\nvectorized batch inference]
    F5 --> G4[Retrieval Service\nTF-IDF cosine similarity over fact-checked claims]
    F6 --> G5[MLflow Service\nquery run metrics + model version history]

    G2 --> H1[Serving Model Bundle\nVectorizer + Logistic Regression + metadata]
    G3 --> H1
    G4 --> H2[Retrieval Bundle\nClaim corpus + TF-IDF vectorizer + matrix]
    G5 --> H3[MLflow Tracking Store + Registry]

    H1 --> I[JSON Response]
    H2 --> I
    H3 --> I
```

---

## Core Components

### 1. Lifespan Startup Loader

The model is loaded **once** at startup and stored in `app.state`.

Startup responsibilities:
- configure MLflow tracking and registry URIs
- load the Production-serving model bundle if available
- train and bootstrap a baseline bundle if no serving artifact exists
- build the retrieval corpus from fact-checked claims
- initialize reusable NLP services

### 2. Logging Middleware

Each request is logged to:
- console
- rotating log file

Logged fields:
- HTTP method
- path
- response status code
- processing time in ms
- client host

### 3. Rate Limiter

Path-specific in-memory rate limits:
- `/classify`: 100 requests/minute
- `/classify/batch`: 10 requests/minute

### 4. Shared NLP Services

Reusable services initialized once:
- `TextCleaner`
- `TokenizerComparer`
- `StopwordManager`
- `StemLemmatizerComparer`

### 5. Serving Model Bundle

The serving bundle contains:
- trained vectorizer
- trained classifier
- label mapping
- model name
- model version
- model stage
- weighted F1
- MLflow run ID
- load timestamp

### 6. Retrieval Bundle

The retrieval bundle contains:
- fact-checked claim corpus
- retrieval TF-IDF vectorizer
- retrieval matrix
- associated labels / sources / metadata

---

## Endpoint Mapping

### `GET /health`
Returns:
- model name
- model version
- stage
- weighted F1
- load timestamp

### `POST /preprocess`
Input:
- text
- processing steps

Returns:
- tokens
- removed stopwords
- processing time

### `POST /classify`
Input:
- text

Returns:
- prediction
- confidence
- class probabilities
- top contributing features

### `POST /classify/batch`
Input:
- up to 100 texts

Returns:
- batch predictions
- total batch processing time

### `POST /retrieve/similar`
Input:
- text
- `top_k`

Returns:
- top-k similar fact-checked claims
- cosine similarity scores

### `GET /model/performance`
Returns live information from MLflow:
- current serving model metrics
- version history
- best family models
- recent run summaries

---

## Validation Rules

### Text input
- minimum length: 10
- maximum length: 10,000

### `top_k`
- minimum: 1
- maximum: 20

### Batch size
- maximum 100 texts per `/classify/batch`

---

## Performance Targets

The architecture is designed to satisfy:

- `/classify < 100ms`
- batch of 10 texts `< 200ms`
- full `/classify/batch` request supports up to 100 texts
- assignment target: full batch under 500ms

This is achieved by:
- loading the model only once at startup
- using shared in-memory vectorizer and classifier objects
- vectorizing the entire batch at once
- avoiding retraining or disk I/O during request handling

---

## Testing Strategy

`test_api.py` will use:
- `pytest`
- `httpx`
- ASGI lifespan-aware integration testing

Coverage includes:
- all six endpoints
- validation failures
- rate-limit behavior
- edge cases
- response-time assertions

---

## Relationship to Task 6

Task 7 depends on Task 6 in two ways:

1. the serving model metadata comes from MLflow-logged experiments and registered models
2. `/model/performance` reads live metrics and version history directly from MLflow

This allows the inference system to expose both prediction functionality and MLOps traceability.
