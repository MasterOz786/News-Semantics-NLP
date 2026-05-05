# Data Storage Architecture Choice

## Selected Storage Option: MinIO-compatible S3

For this assignment, the most appropriate minimal choice is MinIO-compatible object storage. That includes AWS S3-compatible endpoints such as DagsHub S3. The design is intentionally lightweight and portable: the same code works with MinIO, DagsHub, or AWS S3 because they share the S3 API.

## Why MinIO-compatible S3?

- Scalability: Object storage scales horizontally without manual shard management. Data is organized as immutable objects in namespaces, so raw files, processed Parquet assets, and embeddings can all be added without schema migrations or database tuning. A MinIO-compatible server can be deployed locally or in a cloud VM and still support the same scalable access pattern.
- Cost: MinIO-compatible object storage is generally lower cost than managed data warehouse options because it avoids compute-heavy query layers. For an NLP project where the dataset is primarily file-backed and training consumes batches of prepared artifacts, the storage cost is dominated by capacity, not query compute. Using an S3-compatible model lets you keep storage and compute separate; with DagsHub S3, you pay only for the bucket-style object capacity and transfer, not a dedicated database instance.
- Query capability: The storage layer itself is optimized for object retrieval, not row-level SQL queries. That is acceptable here because raw files, Parquet exports, and embeddings are consumed as files. When query capability is required, the same object store can be paired with an analytics layer such as AWS Athena, DagsHub query tools, or Spark on top of Parquet. This architecture keeps the core data lake minimal while still allowing efficient columnar analytics later.

## Three storage layers implemented

1. Raw
   - Stores original ingested files with metadata alongside them.
   - Each dataset version is preserved under `raw/<dataset_name>/<version>/`.
   - Metadata can include source URL, timestamp, and original schema fields.

2. Processed
   - Stores cleaned Parquet files plus derived artifacts.
   - Vocabulary and TF-IDF matrix files are stored in the same versioned namespace.
   - This layer is optimized for machine learning preprocessing and batch training.

3. Embeddings
   - Stores versioned Word2Vec model files under `embeddings/<model_name>/<version>/`.
   - Embeddings are treated as first-class artifacts and can be retrieved independently of raw or processed data.

## Practical fit for DagsHub S3

Because DagsHub exposes an S3-compatible endpoint, the same implementation works without changes to object naming or API behavior. The class is minimal and keeps storage-specific logic in one place, while raw/processed/embeddings are clearly separated. That is the best tradeoff between correctness, portability, and minimal extra complexity.

## Conclusion

A MinIO-compatible S3 design is the minimal correct choice for this assignment. It provides sufficient scalability for dataset growth, low storage-oriented cost, and clean separation of raw, processed, and embeddings artifacts. If query capability is needed, the object store can still back analytic tools like Athena or Spark, while the core Python manager remains simple and robust.

## Working example

The following pattern shows how to instantiate `DataLakeManager` and use it with DagsHub-compatible object storage.

```python
import os
from pathlib import Path
from data_lake_manager import DataLakeManager

manager = DataLakeManager(
    bucket_name=os.environ["DAGSHUB_BUCKET"],
    aws_access_key_id=os.environ["DAGSHUB_ACCESS_KEY"],
    aws_secret_access_key=os.environ["DAGSHUB_SECRET_KEY"],
    endpoint_url=os.environ.get("DAGSHUB_S3_ENDPOINT"),
)

# Raw layer upload
raw_keys = manager.upload_raw(
    dataset_name="covid_fake_news",
    local_paths=[Path("datasets/covid19-fakenews/ClaimFakeCOVID-19_5.csv")],
    metadata={"source": "DagsHub", "version": "v1"},
    version="v1",
)

# Processed layer upload
processed_keys = manager.upload_processed(
    dataset_name="covid_fake_news",
    parquet_paths=[Path("outputs/covid_fake_news_clean.parquet")],
    vocabulary_path=Path("outputs/covid_fake_news_vocab.json"),
    tfidf_path=Path("outputs/covid_fake_news_tfidf.npz"),
    version="v1",
)

# Embeddings layer upload
embedding_keys = manager.upload_embeddings(
    model_name="word2vec",
    version="v1",
    embedding_files=[Path("embeddings/word2vec_v1.bin")],
    metadata={"vector_size": 300},
)

# List versions and fetch artifacts for training
print(manager.list_versions(layer="processed", dataset_name="covid_fake_news"))
downloaded = manager.fetch_for_training(
    layer="processed",
    dataset_name="covid_fake_news",
    version="v1",
    target_dir=Path("downloaded_for_training"),
)
print(downloaded)
```

This example is intentionally minimal, and it demonstrates the three storage layers along with versioned artifact retrieval."