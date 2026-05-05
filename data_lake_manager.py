from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover
    boto3 = None
    ClientError = Exception


class DataLakeManager:
    """Minimal S3-compatible data lake manager for raw, processed, and embeddings storage.

    Designed for MinIO-style object storage and compatible endpoints such as DagsHub S3.
    """

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        s3_client=None,
    ):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.s3 = s3_client or self._create_client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            endpoint_url=endpoint_url,
        )

    def _create_client(
        self,
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        region_name: Optional[str],
        endpoint_url: Optional[str],
    ):
        if boto3 is None:
            raise ImportError(
                "boto3 is required for DataLakeManager. Install it with `pip install boto3`."
            )
        session = boto3.session.Session()
        return session.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            endpoint_url=endpoint_url,
        )

    def _upload_file(self, local_path: Path, key: str, extra_args: Optional[Dict] = None) -> None:
        self.s3.upload_file(str(local_path), self.bucket_name, key, ExtraArgs=extra_args or {})

    def _download_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket_name, key, str(destination))

    def _list_objects(self, prefix: str) -> List[Dict]:
        paginator = self.s3.get_paginator("list_objects_v2")
        result: List[Dict] = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            contents = page.get("Contents") or []
            result.extend(contents)
        return result

    def _list_prefixes(self, prefix: str) -> List[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        prefixes = set()
        for page in paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=prefix,
            Delimiter="/",
        ):
            for common_prefix in page.get("CommonPrefixes", []):
                path = common_prefix.get("Prefix")
                if path:
                    prefixes.add(path)
        return sorted(prefixes)

    def upload_raw(
        self,
        dataset_name: str,
        local_paths: Iterable[Path],
        metadata: Dict,
        version: Optional[str] = None,
    ) -> List[str]:
        """Upload original file artifacts and metadata to the raw storage layer."""
        version = version or "latest"
        prefix = f"raw/{dataset_name}/{version}/"
        uploaded_keys: List[str] = []
        for local_path in local_paths:
            key = prefix + local_path.name
            self._upload_file(local_path, key)
            uploaded_keys.append(key)
        metadata_key = prefix + "metadata.json"
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        uploaded_keys.append(metadata_key)
        return uploaded_keys

    def upload_processed(
        self,
        dataset_name: str,
        parquet_paths: Iterable[Path],
        vocabulary_path: Optional[Path] = None,
        tfidf_path: Optional[Path] = None,
        version: Optional[str] = None,
    ) -> List[str]:
        """Upload cleaned Parquet files, vocabulary, and TF-IDF artifacts to processed storage."""
        version = version or "latest"
        prefix = f"processed/{dataset_name}/{version}/"
        uploaded_keys: List[str] = []
        for local_path in parquet_paths:
            key = prefix + local_path.name
            self._upload_file(local_path, key)
            uploaded_keys.append(key)
        if vocabulary_path is not None:
            vocab_key = prefix + vocabulary_path.name
            self._upload_file(vocabulary_path, vocab_key)
            uploaded_keys.append(vocab_key)
        if tfidf_path is not None:
            tfidf_key = prefix + tfidf_path.name
            self._upload_file(tfidf_path, tfidf_key)
            uploaded_keys.append(tfidf_key)
        return uploaded_keys

    def upload_embeddings(
        self,
        model_name: str,
        version: str,
        embedding_files: Iterable[Path],
        metadata: Optional[Dict] = None,
    ) -> List[str]:
        """Upload versioned Word2Vec embedding files to the embeddings storage layer."""
        prefix = f"embeddings/{model_name}/{version}/"
        uploaded_keys: List[str] = []
        for local_path in embedding_files:
            key = prefix + local_path.name
            self._upload_file(local_path, key)
            uploaded_keys.append(key)
        if metadata is not None:
            metadata_key = prefix + "metadata.json"
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            uploaded_keys.append(metadata_key)
        return uploaded_keys

    def fetch_for_training(
        self,
        dataset_name: Optional[str] = None,
        version: Optional[str] = None,
        layer: str = "processed",
        model_name: Optional[str] = None,
        target_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Download a training-ready set of artifacts for the requested layer."""
        if layer == "embeddings":
            if not model_name:
                raise ValueError("model_name is required for embeddings layer fetch")
            prefix = f"embeddings/{model_name}/{version or 'latest'}/"
        elif layer in {"raw", "processed"}:
            if not dataset_name:
                raise ValueError("dataset_name is required for raw or processed layer fetch")
            prefix = f"{layer}/{dataset_name}/{version or 'latest'}/"
        else:
            raise ValueError("layer must be one of: raw, processed, embeddings")

        destination_root = target_dir or Path.cwd() / "datalake_downloads" / layer
        destination_root.mkdir(parents=True, exist_ok=True)
        objects = self._list_objects(prefix)
        downloaded_paths: List[Path] = []
        for obj in objects:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            relative_key = key[len(prefix) :]
            destination = destination_root / relative_key
            self._download_file(key, destination)
            downloaded_paths.append(destination)
        return downloaded_paths

    def list_versions(
        self,
        layer: str,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[str]:
        """List available version prefixes for a layer."""
        if layer == "embeddings":
            if not model_name:
                raise ValueError("model_name is required for embeddings layer version listing")
            prefix = f"embeddings/{model_name}/"
        elif layer in {"raw", "processed"}:
            if dataset_name:
                prefix = f"{layer}/{dataset_name}/"
            else:
                prefix = f"{layer}/"
        else:
            raise ValueError("layer must be one of: raw, processed, embeddings")

        prefixes = self._list_prefixes(prefix)
        versions = [p[len(prefix) :].rstrip("/") for p in prefixes]
        return sorted(set(versions))


if __name__ == "__main__":
    import os

    bucket_name = os.environ.get("DAGSHUB_BUCKET", "your-dagshub-bucket")
    access_key = os.environ.get("DAGSHUB_ACCESS_KEY")
    secret_key = os.environ.get("DAGSHUB_SECRET_KEY")
    endpoint_url = os.environ.get("DAGSHUB_S3_ENDPOINT")

    manager = DataLakeManager(
        bucket_name=bucket_name,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
    )

    print("Example DataLakeManager usage")
    print("  bucket:", bucket_name)

    raw_files = [Path("datasets/covid19-fakenews/ClaimFakeCOVID-19_5.csv")]
    metadata = {"source": "DagsHub example", "version": "v1"}
    print("Uploading raw files...")
    try:
        raw_keys = manager.upload_raw(
            dataset_name="covid_fake_news",
            local_paths=raw_files,
            metadata=metadata,
            version="v1",
        )
        print("Uploaded raw keys:", raw_keys)
    except Exception as exc:
        print("Raw upload failed:", exc)

    print("Listing processed versions...")
    try:
        versions = manager.list_versions(layer="processed", dataset_name="covid_fake_news")
        print("Processed versions:", versions)
    except Exception as exc:
        print("List versions failed:", exc)
