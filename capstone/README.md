# 📊 Sentiment Analysis Dashboard
### Capstone Project — PKCERT AI & Software Development Internship (Task 28: Final Task)

This project is a complete, end-to-end sentiment analysis system that brings together
everything learned during the internship — from data preparation, building and
evaluating an NLP model, to developing the Backend API, Frontend UI, testing,
and deploying the system.

Users can type in a review (in Thai or English) and the system will analyze whether
the text is Positive, Negative, or Neutral, along with a confidence score for
each class.

---

## Table of Contents
1. [Project Overview (Part A)](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset and Data Preparation (Part B)](#3-dataset-and-data-preparation)
4. [Model Building and Evaluation (Part B)](#4-model-building-and-evaluation)
5. [Backend API (Part C)](#5-backend-api)
6. [Frontend Dashboard (Part C)](#6-frontend-dashboard)
7. [System Testing (Part D)](#7-system-testing)
8. [How to Install and Run the Project](#8-how-to-install-and-run-the-project)
9. [Deploying with Docker](#9-deploying-with-docker)
10. [Full File Structure](#10-full-file-structure)
11. [Limitations and Future Improvements](#11-limitations-and-future-improvements)

---

## 1. Project Overview

**Problem Statement:** Customer service/marketing teams need to read large volumes
of product reviews by hand to gauge customer satisfaction, which is slow and
inconsistent.

**Target Users:** Owners of small-to-medium online stores, or customer service
teams who need a quick way to screen the sentiment of reviews before doing
deeper analysis.

**Value Delivered:** Real-time sentiment analysis of text in both Thai and
English, through an easy-to-use web page, cutting down the time spent reading
reviews manually.

**Project Scope**
| In-scope | Out-of-scope |
|---|---|
| 3-class sentiment analysis (positive/negative/neutral) | User account / login system |
| Supports Thai + English text | Audio/image analysis |
| Single-text and batch analysis | Persistent history storage in a database |
| Dashboard display with confidence charts | Support for languages other than Thai/English |
| Deploy via Docker | Separate mobile application |

**Technology Stack Chosen and Rationale**
| Layer | Technology | Rationale |
|---|---|---|
| Data / Model | Python, pandas, scikit-learn (TF-IDF + Logistic Regression) | Builds on Task 27, which compared static vs. contextual embeddings — TF-IDF was chosen because the deployment environment has no access to the Hugging Face Hub, and a small, fast-inference model suited to CPU was needed |
| Backend/API | FastAPI + Uvicorn | Type-safe with Pydantic, automatic docs (/docs), good performance, and consistent with the full-stack pattern practiced in previous work |
| Frontend/UI | HTML + CSS + Vanilla JavaScript (fetch API) | No build tool required, making deployment simple and lightweight — well suited to a small dashboard focused on fast delivery |
| Deployment | Docker + docker-compose | Reproducible runs, with backend/frontend clearly separated into their own containers |

**Execution Plan (steps going forward)**
1. Prepare data → train model (data readiness must be complete before model training)
2. Evaluate/improve the model (model readiness must be complete before API integration)
3. Build the Backend API that loads the trained model
4. Build the Frontend and connect it to the Backend
5. Test the whole system, fix bugs, and clean up the code
6. Deploy with Docker and write accompanying documentation

**Risks and Mitigation Strategies**
- *Too little data* → Use synthetic data that is varied enough to demonstrate the pipeline; clearly noted that a real production system should use a much larger real dataset.
- *No internet access to load a pretrained transformer* → Use TF-IDF + Logistic Regression instead, which still delivers good performance for 3-class sentiment on a small-to-medium dataset.
- *CORS between frontend/backend on different origins* → Enable CORS middleware in FastAPI.

**Success Criteria / Evaluation Metrics (revisited in Part D)**
- Accuracy on the test set ≥ 70%
- F1-score (macro) ≥ 0.70
- API latency per request < 50 ms (excluding network latency)
- Users can analyze text and see results within 3 clicks

---

## 2. System Architecture

See the full architecture diagram at [`docs/architecture.svg`](docs/architecture.svg)
(or the Mermaid diagram file at [`docs/architecture.mmd`](docs/architecture.mmd))

```
[CSV Dataset] --> [train_model.py: TF-IDF + LogisticRegression] --> [sentiment_model.joblib]
                                                                            |
                                                                            v
                                                              [FastAPI Backend :8000]
                                                          /api/predict  /api/health  /api/metrics
                                                                            |
                                                                            v
                                                          [HTML/JS Dashboard :8080] <-- User
```

---

## 3. Dataset and Data Preparation

- **Data Source:** A synthetic dataset generated in-house by `data/generate_dataset.py`,
  simulating product/service reviews in both Thai and English across 3 classes
  (positive/negative/neutral). Since the development environment has no internet
  access to public dataset hubs, a synthetic dataset resembling real-world data
  was created instead of using an external dataset (no licensing concerns, since
  the data is self-generated).
- **Data Cleaning:** Removes duplicate whitespace (`clean_text()` in
  `model/train_model.py`), drops rows with empty text.
- **Data Split:** Stratified split of 70% train / 15% validation / 15% test to
  keep the proportion of each class equal across all sets — well suited to
  classification tasks with a limited number of samples per class, to prevent
  any one split from missing examples of a given class entirely.

Run this command to regenerate the dataset:
```bash
python3 data/generate_dataset.py
```

---

## 4. Model Building and Evaluation

**Approach Chosen:** TF-IDF (unigram/bigram) + Logistic Regression (multi-class)
instead of fine-tuning a Transformer model as in Task 27, due to network
connectivity limitations in the deployment environment (no access to the
Hugging Face Hub), and to get a model with fast inference and a small file
size, suited to running on CPU/serverless, without a major hit to accuracy
for 3-class sentiment on short text.

**Training Pipeline** — see the full code in `model/train_model.py`
1. Load and clean the data
2. Split into train/val/test
3. Train 2 iterations to compare hyperparameters:
   - **Iteration 1 (baseline):** `ngram_range=(1,1)`, `C=1.0`
   - **Iteration 2 (tuned):** `ngram_range=(1,2)`, `C=5.0`
4. Select the model with the best F1 on the validation set, saved as `model/sentiment_model.joblib`

**Evaluation Results (actual results from the latest run, recorded in `model/metrics.json`)**

| Metric | Iteration 1 (baseline) | Iteration 2 (tuned, selected) |
|---|---|---|
| Validation F1 (macro) | 0.8648 | 0.8648 |
| Test F1 (macro) | 0.9666 | 0.9666 |
| Test Accuracy | — | 0.9667 |

The selected model (Iteration 2) meets the success criteria set out in Part A (Accuracy ≥ 70%, F1 ≥ 0.70).

**Error Analysis:** In the most recent run, 1 out of 30 test samples was
mispredicted — a case where the sentence began with a connecting phrase
("Honestly, ...") before a positive statement, which confused the model due
to a word pattern that was rare in the training set. This is an expected
limitation of TF-IDF, which does not understand sentence context as well as
contextual embeddings (e.g., BERT in Task 27).

Run this command to retrain the model:
```bash
python3 model/train_model.py
```

---

## 5. Backend API

The main code is in `backend/main.py` (FastAPI) — once running, the automatic
API docs can be viewed at `http://localhost:8000/docs`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Checks server status and whether the model loaded successfully |
| POST | `/api/predict` | Analyzes the sentiment of a single text (`{"text": "..."}`) |
| POST | `/api/predict/batch` | Analyzes the sentiment of multiple texts at once (`{"texts": [...]}`) |
| GET | `/api/metrics` | Returns the model's latest performance metrics |

**Example Request/Response**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The product is great and easy to use"}'
```
```json
{
  "text": "The product is great and easy to use",
  "label": "positive",
  "confidence": 0.6928,
  "probabilities": {"negative": 0.1482, "neutral": 0.159, "positive": 0.6928},
  "latency_ms": 1.23
}
```

**Input validation:** Uses Pydantic to ensure `text` is not empty and is no
longer than 2000 characters. If validation fails, it returns HTTP 422 with
error details.

**On Authentication:** This project is a public analysis tool that does not
store any personal data or user accounts, so it was decided that **a
login/JWT system is not required** in this version (as stated in the Part A
scope). If persistent per-user search history is needed in the future, a JWT
system issued via an HTTP-only cookie, with protected routes via middleware,
would be added.

---

## 6. Frontend Dashboard

A single self-contained file at `frontend/index.html` (HTML + CSS + Vanilla JS, no build tool required)

Main features:
- A text input box with an "Analyze Sentiment" button
- Displays results as a colored badge + probability bar chart for each class (positive/neutral/negative)
- Shows the backend connection status (online/offline) in real time
- Loading state while calling the API, and an error state when the connection fails
- Recent analysis history (kept in the page's memory during the session)
- A panel showing Model Performance metrics (pulled from `/api/metrics`)
- A field to set the Backend API URL manually (supports cases where the backend is deployed separately from the frontend)

---

## 7. System Testing

Test files are in `tests/test_api.py`, using `pytest` + the FastAPI `TestClient`,
covering the following test cases (all 8/8 passing on the actual run):

| Test Case | Expected Result |
|---|---|
| `test_health_check` | API responds with status ok and the model loaded successfully |
| `test_predict_positive_text` | Positive text is classified as positive |
| `test_predict_negative_text_thai` | Negative Thai text is classified as negative |
| `test_predict_empty_text_rejected` | Empty text is rejected with a 422 |
| `test_predict_missing_field` | Missing the text field returns a 422 |
| `test_batch_predict` | Multiple texts are analyzed correctly at once |
| `test_batch_predict_filters_blank_entries` | Blank entries in a batch list are skipped without failing the request |
| `test_metrics_endpoint` | Returned metrics meet the accuracy ≥ 70% criterion set in Part A |

Run the tests:
```bash
pip install -r backend/requirements.txt
pip install pytest httpx2
pytest tests/test_api.py -v
```

**Bug found and fixed during development:** Found that a newer version of
`sklearn.linear_model.LogisticRegression` removed the `multi_class` parameter
(deprecated), which caused the training script to error out — fixed by
removing that parameter and letting sklearn automatically choose the
multi-class strategy.

---

## 8. How to Install and Run the Project

### System Requirements
- Python 3.10 or higher
- pip
- (Optional) Docker + Docker Compose for deployment

### Steps to Run Locally (without Docker)

```bash
# 1) Install dependencies
pip install -r backend/requirements.txt

# 2) Generate the dataset (if data/reviews_dataset.csv doesn't exist yet)
python3 data/generate_dataset.py

# 3) Train the model (produces model/sentiment_model.joblib and model/metrics.json)
python3 model/train_model.py

# 4) Run the Backend API
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 5) Open frontend/index.html directly in a browser
#    or run a simple static web server:
cd frontend && python3 -m http.server 8080
# then open http://localhost:8080
```

> Note: The web page has a field to set the "Backend API URL" — the default is
> `http://127.0.0.1:8000`, and it can be adjusted to match wherever the backend is actually running.

---

## 9. Deploying with Docker

```bash
docker compose up --build
```

- Backend runs at `http://localhost:8000`
- Frontend (nginx) runs at `http://localhost:8080`

**Relevant environment variables:** Currently the system does not require any
additional environment variables for basic operation (there is no database or
secret key to configure). To change the backend URL the frontend calls, use
the "Backend API URL" field on the web page directly — no rebuild needed.

**Files relevant to deployment:**
- `backend/Dockerfile` — builds the image for the FastAPI backend
- `frontend/Dockerfile` — builds the nginx image that serves the static files
- `docker-compose.yml` — orchestrates both services together

---

## 10. Full File Structure

```
capstone/
├── README.md                     # This document
├── docker-compose.yml            # Runs the whole system with Docker
├── data/
│   ├── generate_dataset.py       # Generates the synthetic dataset
│   └── reviews_dataset.csv       # The dataset used for training (generated by the script above)
├── model/
│   ├── train_model.py            # Trains + evaluates + saves the model
│   ├── sentiment_model.joblib    # The trained model (TF-IDF + LogisticRegression)
│   └── metrics.json              # Model evaluation results (before/after iteration)
├── backend/
│   ├── main.py                   # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   └── Dockerfile
├── frontend/
│   ├── index.html                # Dashboard (single HTML/CSS/JS file)
│   └── Dockerfile
├── tests/
│   └── test_api.py               # Automated tests (pytest)
└── docs/
    ├── architecture.svg          # System architecture diagram
    └── architecture.mmd          # Mermaid diagram (easy to edit)
```

---

## 11. Limitations and Future Improvements

- **Data:** A small synthetic dataset (~200 samples), suitable for
  demonstrating the pipeline. Real-world use should collect a much larger
  volume of real review data for more robust accuracy.
- **Model:** Uses TF-IDF + Logistic Regression instead of a Transformer due to
  network limitations in the development environment — in a real production
  environment with internet access, it's recommended to try fine-tuning a
  multilingual Transformer model (following the Task 27 approach) to better
  capture sentence context, especially sentences with connecting phrases,
  sarcasm, or double negatives.
- **Search history:** Currently kept only in browser memory (lost on refresh).
  Persistent storage would require adding a database and an authentication
  system, as noted in Part C.

---

*This project was created as the Capstone Project (Task 28) for the
PKCERT AI & Software Development Internship program*
