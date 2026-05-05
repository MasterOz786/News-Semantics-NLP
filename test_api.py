import time

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app import app


@pytest_asyncio.fixture(scope="module")
async def client():
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as async_client:
            yield async_client


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "stage" in data
    assert "weighted_f1" in data
    assert "load_timestamp" in data


@pytest.mark.asyncio
async def test_preprocess_endpoint(client):
    payload = {
        "text": "This is a test article about a fact checked political claim.",
        "steps": {
            "tokenizer": "custom",
            "stopword_list": "custom",
            "normalization": "lemmatizer",
            "min_token_length": 2,
        },
    }
    response = await client.post("/preprocess", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["tokens"], list)
    assert isinstance(data["removed_stopwords"], list)
    assert data["processing_time_ms"] >= 0


@pytest.mark.asyncio
async def test_classify_endpoint(client):
    payload = {
        "text": "Reuters reported that officials denied the false claim spreading online."
    }
    start = time.perf_counter()
    response = await client.post("/classify", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "class_probabilities" in data
    assert "top_contributing_features" in data
    assert elapsed_ms < 100


@pytest.mark.asyncio
async def test_classify_batch_endpoint(client):
    texts = [
        "Reuters reported that officials denied the false claim spreading online.",
        "A viral post claimed a fabricated conspiracy without evidence from experts.",
        "Fact checkers reviewed the statement and found major context missing.",
        "Officials said the headline circulating on social media was misleading.",
        "Researchers published updated findings that contradict the fake narrative.",
        "The post exaggerated the event and misrepresented the original report.",
        "The article cited verified sources and included formal attribution language.",
        "A manipulated clip was shared online with a misleading political caption.",
        "Independent reporting confirmed the claim was false after investigation.",
        "The message reused sensational wording common in fabricated stories online.",
    ]
    start = time.perf_counter()
    response = await client.post("/classify/batch", json={"texts": texts})
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert len(data["results"]) == 10
    assert elapsed_ms < 200


@pytest.mark.asyncio
async def test_retrieve_similar_endpoint(client):
    payload = {
        "text": "A politician made a misleading statement that was later fact checked.",
        "top_k": 5,
    }
    response = await client.post("/retrieve/similar", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["top_k"] == 5
    assert len(data["results"]) <= 5
    if data["results"]:
        assert "claim" in data["results"][0]
        assert "cosine_similarity" in data["results"][0]


@pytest.mark.asyncio
async def test_model_performance_endpoint(client):
    response = await client.get("/model/performance")
    assert response.status_code == 200
    data = response.json()
    assert "current_model" in data
    assert "version_history" in data
    assert "recent_runs" in data


@pytest.mark.asyncio
async def test_text_validation_error(client):
    response = await client.post("/classify", json={"text": "short"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_top_k_validation_error(client):
    response = await client.post(
        "/retrieve/similar",
        json={
            "text": "This is a valid enough retrieval request for testing.",
            "top_k": 25,
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_size_validation_error(client):
    texts = ["This is a sufficiently long batch item for validation testing."] * 101
    response = await client.post("/classify/batch", json={"texts": texts})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_rate_limit(client):
    payload = {
        "texts": [
            "This is a sufficiently long request item for testing the batch limiter."
        ]
    }
    headers = {"x-forwarded-for": "rate-limit-test-client"}
    for _ in range(10):
        response = await client.post("/classify/batch", json=payload, headers=headers)
        assert response.status_code == 200
    response = await client.post("/classify/batch", json=payload, headers=headers)
    assert response.status_code == 429
