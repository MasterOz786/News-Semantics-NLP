# Task 7 — FastAPI Screenshots Guide and Placeholders

This file is a **submission-ready screenshot guide** for Task 7.

It includes:
- what to run
- what to capture
- where to capture it from
- suggested screenshot filenames
- ready-to-fill Markdown image placeholders
- suggested captions for the final report

---

## Goal

The assignment requires a FastAPI inference system with:
- architecture diagram before coding
- six API endpoints
- request validation
- request logging middleware
- model loading at startup using lifespan
- rate limiting
- pytest + httpx test coverage
- response-time assertions

This guide helps you produce clean evidence screenshots for the final report.

---

## Before taking screenshots

### Step 1 — Make sure the API files exist

These files should already be present:
- `app.py`
- `test_api.py`
- `TASK_7_FASTAPI_ARCHITECTURE.md`

### Step 2 — Start the API

From the `2/` directory, run one of these:

- `venv311/bin/python app.py`

or

- `venv311/bin/uvicorn app:app --host 127.0.0.1 --port 8000`

### Step 3 — Open the API locally

Use:
- `http://127.0.0.1:8000`

Optional interactive docs:
- `http://127.0.0.1:8000/docs`

### Step 4 — Run the tests

From the `2/` directory, run:

- `venv311/bin/pytest -q test_api.py`

### Step 5 — Save screenshots here

Store all screenshots in:

- `2/screenshots_api/`

---

## Screenshot Checklist

Below are **15 recommended screenshots** for Task 7.

---

## Screenshot 1 — Task 7 architecture diagram

**What to capture:**
- the architecture diagram from `TASK_7_FASTAPI_ARCHITECTURE.md`
- include the request flow and startup flow if possible

**Where from:**
- open `TASK_7_FASTAPI_ARCHITECTURE.md`

**Suggested filename:**
- `screenshots_api/api_01_architecture_diagram.png`

**Suggested caption:**
- Task 7 FastAPI system architecture diagram showing startup, routing, inference, retrieval, and MLflow integration.

![Task 7 Screenshot 01 Placeholder](screenshots_api/api_01_architecture_diagram.png)

---

## Screenshot 2 — API startup terminal

**What to capture:**
- terminal showing the app starting successfully
- if possible, include the model loaded message with model name, version, stage, and F1

**Where from:**
- terminal after running `app.py` or `uvicorn`

**Suggested filename:**
- `screenshots_api/api_02_startup_terminal.png`

**Suggested caption:**
- FastAPI startup log showing successful model loading through the lifespan context manager.

![Task 7 Screenshot 02 Placeholder](screenshots_api/api_02_startup_terminal.png)

---

## Screenshot 3 — FastAPI Swagger/OpenAPI docs page

**What to capture:**
- the `/docs` page showing the available endpoints

**Where from:**
- `http://127.0.0.1:8000/docs`

**Suggested filename:**
- `screenshots_api/api_03_docs_page.png`

**Suggested caption:**
- FastAPI interactive documentation showing the implemented inference endpoints.

![Task 7 Screenshot 03 Placeholder](screenshots_api/api_03_docs_page.png)

---

## Screenshot 4 — GET /health response

**What to capture:**
- the `/health` endpoint response
- should show:
  - model name
  - model version
  - stage
  - F1 score
  - load timestamp

**Where from:**
- Swagger UI or Postman or browser response view

**Suggested filename:**
- `screenshots_api/api_04_health_response.png`

**Suggested caption:**
- Health endpoint response showing model identity, stage, F1 score, and load timestamp.

![Task 7 Screenshot 04 Placeholder](screenshots_api/api_04_health_response.png)

---

## Screenshot 5 — POST /preprocess request and response

**What to capture:**
- request payload and response for `/preprocess`
- response should show:
  - tokens
  - removed stopwords
  - processing time

**Where from:**
- Swagger UI or Postman

**Suggested filename:**
- `screenshots_api/api_05_preprocess_response.png`

**Suggested caption:**
- Preprocess endpoint example showing tokens, removed stopwords, and processing time.

![Task 7 Screenshot 05 Placeholder](screenshots_api/api_05_preprocess_response.png)

---

## Screenshot 6 — POST /classify request and response

**What to capture:**
- request payload and response for `/classify`
- response should show:
  - prediction
  - confidence
  - class probabilities
  - top contributing features

**Where from:**
- Swagger UI or Postman

**Suggested filename:**
- `screenshots_api/api_06_classify_response.png`

**Suggested caption:**
- Classify endpoint output showing prediction, confidence, class probabilities, and top contributing features.

![Task 7 Screenshot 06 Placeholder](screenshots_api/api_06_classify_response.png)

---

## Screenshot 7 — POST /classify/batch request and response

**What to capture:**
- request payload with multiple texts
- response showing batch predictions and total processing time

**Where from:**
- Swagger UI or Postman

**Suggested filename:**
- `screenshots_api/api_07_batch_classify_response.png`

**Suggested caption:**
- Batch classify endpoint output showing multiple predictions and total batch processing time.

![Task 7 Screenshot 07 Placeholder](screenshots_api/api_07_batch_classify_response.png)

---

## Screenshot 8 — POST /retrieve/similar request and response

**What to capture:**
- request payload and response for `/retrieve/similar`
- response should show:
  - returned claims
  - labels
  - sources
  - cosine similarity scores

**Where from:**
- Swagger UI or Postman

**Suggested filename:**
- `screenshots_api/api_08_retrieve_similar_response.png`

**Suggested caption:**
- Retrieve similar endpoint output showing top-k fact-checked claims with cosine similarity scores.

![Task 7 Screenshot 08 Placeholder](screenshots_api/api_08_retrieve_similar_response.png)

---

## Screenshot 9 — GET /model/performance response

**What to capture:**
- `/model/performance` response
- should show:
  - current model metrics
  - version history
  - recent runs

**Where from:**
- Swagger UI or browser or Postman

**Suggested filename:**
- `screenshots_api/api_09_model_performance_response.png`

**Suggested caption:**
- Model performance endpoint showing live metrics and version history pulled from MLflow.

![Task 7 Screenshot 09 Placeholder](screenshots_api/api_09_model_performance_response.png)

---

## Screenshot 10 — Request logging in terminal

**What to capture:**
- terminal log lines showing incoming API requests
- include method, path, status code, client, and timing if visible

**Where from:**
- terminal running the app while requests are being made

**Suggested filename:**
- `screenshots_api/api_10_request_logging_terminal.png`

**Suggested caption:**
- Console request logging produced by the API middleware.

![Task 7 Screenshot 10 Placeholder](screenshots_api/api_10_request_logging_terminal.png)

---

## Screenshot 11 — Rotating file log evidence

**What to capture:**
- the generated log file, or a file viewer showing recent API request lines
- ideally show `logs/app.log`

**Where from:**
- Finder / file explorer or text editor view of `logs/app.log`

**Suggested filename:**
- `screenshots_api/api_11_rotating_log_file.png`

**Suggested caption:**
- Rotating file log showing persisted API request records.

![Task 7 Screenshot 11 Placeholder](screenshots_api/api_11_rotating_log_file.png)

---

## Screenshot 12 — Validation error example

**What to capture:**
- an invalid request example, such as:
  - text too short
  - `top_k` greater than 20
  - batch size greater than 100
- response should show validation failure

**Where from:**
- Swagger UI or Postman

**Suggested filename:**
- `screenshots_api/api_12_validation_error.png`

**Suggested caption:**
- Validation failure example demonstrating Pydantic input enforcement.

![Task 7 Screenshot 12 Placeholder](screenshots_api/api_12_validation_error.png)

---

## Screenshot 13 — Rate limit evidence

**What to capture:**
- a request that returns a `429` rate-limit response
- ideally for `/classify/batch`

**Where from:**
- test output, Swagger UI, Postman, or terminal-assisted repeated requests

**Suggested filename:**
- `screenshots_api/api_13_rate_limit_response.png`

**Suggested caption:**
- Rate limiting evidence showing a `429` response after exceeding the configured request threshold.

![Task 7 Screenshot 13 Placeholder](screenshots_api/api_13_rate_limit_response.png)

---

## Screenshot 14 — Pytest output showing all tests passed

**What to capture:**
- terminal after running:
  - `venv311/bin/pytest -q test_api.py`
- capture the line showing the tests passed

**Where from:**
- terminal output

**Suggested filename:**
- `screenshots_api/api_14_pytest_passed.png`

**Suggested caption:**
- Pytest execution showing all API tests passed successfully.

![Task 7 Screenshot 14 Placeholder](screenshots_api/api_14_pytest_passed.png)

---

## Screenshot 15 — Response-time assertion evidence

**What to capture:**
- either:
  - pytest output including response-time tests, or
  - terminal/Postman timing evidence for `/classify` and batch of 10 texts

**Where from:**
- pytest terminal output or request client timing panel

**Suggested filename:**
- `screenshots_api/api_15_response_time_evidence.png`

**Suggested caption:**
- Response-time evidence showing the API satisfies the required latency assertions.

![Task 7 Screenshot 15 Placeholder](screenshots_api/api_15_response_time_evidence.png)

---

## Optional extra screenshots

If you want stronger evidence, add these too:

### Optional Screenshot 16 — Finder/file explorer showing Task 7 files
- `screenshots_api/api_16_files_overview.png`

### Optional Screenshot 17 — `/docs` expanded endpoint details
- `screenshots_api/api_17_docs_endpoint_detail.png`

### Optional Screenshot 18 — Terminal showing MLflow-backed model loading
- `screenshots_api/api_18_mlflow_model_loading.png`

### Optional Screenshot 19 — `/model/performance` expanded version history section
- `screenshots_api/api_19_model_version_history.png`

### Optional Screenshot 20 — Batch classify large input example
- `screenshots_api/api_20_large_batch_example.png`

---

## Embedding instructions for the final report

For each screenshot in the report:
1. place the image under the relevant Task 7 subsection
2. add a one-line caption
3. reference the figure in the explanation text

Example style:
- Figure X shows the health endpoint exposing model identity and load timestamp.
- Figure Y shows the classify endpoint returning prediction confidence and top contributing features.

---

## Quick command summary

From the `2/` directory:

### Start the app
- `venv311/bin/python app.py`

or

- `venv311/bin/uvicorn app:app --host 127.0.0.1 --port 8000`

### Open docs
- `http://127.0.0.1:8000/docs`

### Run tests
- `venv311/bin/pytest -q test_api.py`

---

## Final checklist

Before submission, confirm that:
- all screenshots are saved in `2/screenshots_api/`
- the architecture diagram screenshot is included
- all six endpoints are represented in screenshots
- validation evidence is included
- logging evidence is included
- rate limiting evidence is included
- pytest success evidence is included
- response-time evidence is included

---

## Notes

If the app does not open:
- verify the API is running on port `8000`
- verify you launched it from the `2/` directory
- check the terminal for import or startup errors

If you want cleaner screenshots, use the Swagger UI for endpoint requests and responses, and the terminal for logs/test output.
