# Containerized ML/NLP Pipeline Guide

This guide explains how the Docker and DVC pipeline integration works for the project.

---

## What is containerized

The project now includes:

- `Dockerfile`
- `docker-compose.yml`
- DVC stages that use Docker Compose

This means the ML/NLP pipeline can now be executed in a containerized and reproducible way.

---

## Services in Docker Compose

### 1. `api`
Runs the FastAPI inference system and can also be used as a general execution container for:
- Task 6 MLflow runs
- Task 7 pytest execution
- general Python project commands

### 2. `mlflow-ui`
Runs the MLflow tracking UI so experiment runs and registry entries can be inspected in the browser.

---

## DVC stages added

### `build_container_stack`
Builds the Docker images used by the project.

### `task6_mlflow_containerized`
Runs Task 6 inside the `api` container:
- executes `python task6_mlflow.py`
- generates MLflow experiment runs
- writes a completion marker to `pipeline_artifacts/task6_mlflow_containerized.txt`

### `task7_api_tests_containerized`
Runs Task 7 tests inside the `api` container:
- executes `pytest -q test_api.py`
- writes a completion marker to `pipeline_artifacts/task7_api_tests_containerized.txt`

---

## How to run the containerized pipeline

From the `2/` directory:

### Build the containers
- `docker compose build`

### Run full DVC pipeline
- `dvc repro`

### Run only Task 6 stage
- `dvc repro task6_mlflow_containerized`

### Run only Task 7 containerized tests
- `dvc repro task7_api_tests_containerized`

---

## How to run services manually

### Start FastAPI app
- `docker compose up api`

API will be available at:
- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

### Start MLflow UI
- `docker compose up mlflow-ui`

MLflow will be available at:
- `http://127.0.0.1:5001`

### Start both together
- `docker compose up`

---

## Pipeline outputs

When the containerized DVC stages finish, these marker files are created:

- `pipeline_artifacts/task6_mlflow_containerized.txt`
- `pipeline_artifacts/task7_api_tests_containerized.txt`

These act as simple pipeline completion outputs for DVC tracking.

---

## Why this integration helps

This setup ensures:
- reproducibility across environments
- consistent dependency management
- containerized execution of MLflow experiments
- containerized verification of FastAPI endpoints
- easier submission evidence for deployment and MLOps workflow

---

## Suggested submission wording

You can describe the integration like this:

- The ML/NLP workflow was integrated with Docker Compose and DVC so that Task 6 MLflow experiment tracking and Task 7 FastAPI API tests can be executed in a reproducible containerized pipeline.

---

## Recommended evidence screenshots

If you want screenshots for this integration, capture:

1. `docker compose build` terminal output
2. `dvc repro` terminal output
3. `docker compose up` showing both services starting
4. FastAPI docs page from container
5. MLflow UI page from container
6. presence of `pipeline_artifacts/` marker files
