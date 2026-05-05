from __future__ import annotations

import os
from pathlib import Path

from data_lake_manager import DataLakeManager


def main() -> None:
    # Environment variables are the cleanest way to keep credentials out of source.
    bucket_name = os.environ.get("DAGSHUB_BUCKET") or "your-dagshub-bucket"
    access_key = os.environ.get("DAGSHUB_ACCESS_KEY")
    secret_key = os.environ.get("DAGSHUB_SECRET_KEY")
    endpoint_url = os.environ.get("DAGSHUB_S3_ENDPOINT")

    manager = DataLakeManager(
        bucket_name=bucket_name,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
    )

    raw_files = [Path("datasets/covid19-fakenews/ClaimFakeCOVID-19_5.csv")]
    raw_metadata = {
        "source": "DagsHub dataset upload",
        "description": "Original COVID-19 fake news claims dataset",
        "version": "v1",
    }
    print("Uploading raw artifacts...")
    raw_keys = manager.upload_raw(
        dataset_name="covid_fake_news",
        local_paths=raw_files,
        metadata=raw_metadata,
        version="v1",
    )
    print("Raw upload complete:", raw_keys)

    processed_files = [Path("outputs/covid_fake_news_clean.parquet")]
    vocabulary_file = Path("outputs/covid_fake_news_vocab.json")
    tfidf_file = Path("outputs/covid_fake_news_tfidf.npz")
    print("Uploading processed artifacts...")
    processed_keys = manager.upload_processed(
        dataset_name="covid_fake_news",
        parquet_paths=processed_files,
        vocabulary_path=vocabulary_file,
        tfidf_path=tfidf_file,
        version="v1",
    )
    print("Processed upload complete:", processed_keys)

    embedding_files = [Path("embeddings/word2vec_v1.bin")]
    print("Uploading embeddings...")
    embedding_keys = manager.upload_embeddings(
        model_name="word2vec",
        version="v1",
        embedding_files=embedding_files,
        metadata={"vector_size": 300, "window": 5},
    )
    print("Embeddings upload complete:", embedding_keys)

    print("Available processed versions:")
    versions = manager.list_versions(layer="processed", dataset_name="covid_fake_news")
    print(versions)

    print("Downloading processed layer for training...")
    downloaded = manager.fetch_for_training(
        layer="processed",
        dataset_name="covid_fake_news",
        version="v1",
        target_dir=Path("downloaded_for_training"),
    )
    print("Downloaded files:", downloaded)


if __name__ == "__main__":
    main()
