# Project plan - weekly roadmap 

## Week 1 — Project setup & data intake

* Initialize Git repo and README (project goals, MVP scope).
* Create `docker-compose` skeleton (Postgres, backend, optional Redis).
* Add raw dataset to `data/`.
  **Deliverable:** repo skeleton + initial dataset + README.

## Week 2 — Basic DRF skeleton & auth

* Scaffold Django + DRF project and apps (profiles, records, predictions).
* Implement user registration and JWT auth.
  **Deliverable:** working auth endpoints and basic API structure.

## Week 3 — Exploratory Data Analysis (EDA) — part 1

* Global distributions, missing value inventory, basic visualizations.
* Define target labels for multi-task outputs.
  **Deliverable:** `01_EDA.ipynb` with summary findings.

## Week 4 — EDA & preprocessing — part 2

* Feature engineering (age buckets, interactions, encoding).
* Train/validation/test split and preprocessing pipelines.
  **Deliverable:** cleaned dataset and preprocessing scripts.

## Week 5 — Baseline models (per-task)

* Train simple baselines (Logistic Regression / LightGBM) for each task (Sleep_Quality, Stress_Level, Health_Issues).
* Evaluate basic performance and save baseline metrics.
  **Deliverable:** `02_baselines.ipynb` + saved baseline models.

## Week 6 — Model packaging & inference function

* Wrap model inference into a reusable function/module (single interface).
* Save models (joblib) and create `model_version` metadata.
  **Deliverable:** inference module + model artifacts.

## Week 7 — Prediction API endpoint

* Implement `POST /api/predict/state/` in DRF using inference module.
* Return classes, probabilities, and prediction_id.
  **Deliverable:** working predict endpoint + API test.

## Week 8 — Clustering: pipeline & analysis

* Build clustering pipeline (PCA/UMAP + KMeans or HDBSCAN).
* Profile clusters (descriptive statistics, country patterns).
  **Deliverable:** `03_clustering.ipynb` + cluster assignments.

## Week 9 — Cluster API & minimal UI view

* Implement `/api/cluster/` and endpoint to fetch cluster profile for a user.
* Add basic Streamlit page showing UMAP + cluster profile.
  **Deliverable:** cluster endpoint + Streamlit cluster visualization.

## Week 10 — SHAP: integration POC

* Compute SHAP values for baseline models; create summary plots.
* Design how SHAP output will be returned (JSON or HTML snippet).
  **Deliverable:** `04_shap.ipynb` + sample SHAP outputs.

## Week 11 — Explainability endpoint & Streamlit explain view

* Add explainability to predict response or separate `/api/explain/{prediction_id}/`.
* Streamlit: show SHAP summary and force/decision plots.
  **Deliverable:** explain endpoint + Streamlit explain panel.

## Week 12 — Improve models / calibration & cross-validation

* Run cross-validation, tune models (simple hyperparameter search), calibrate probabilities.
* Update saved models and model_version records.
  **Deliverable:** improved model metrics and updated artifacts.

## Week 13 — Recommendation POC: simple rule-based + uplift baseline design

* Implement a simple rule-based recommender (safety-first) as fallback.
* Prototype uplift idea (two-model approach or simple CATE baseline).
  **Deliverable:** rule-based recommender + uplift baseline notebook.

## Week 14 — Recommendation API & Streamlit view (POC)

* Implement `/api/recommendation/` returning recommended cups + expected effects (POC).
* Streamlit: show recommendation with brief explanation & SHAP-backed rationale.
  **Deliverable:** recommendation endpoint + UI.

## Week 15 — Anomaly detection (POC)

* Build an IsolationForest or simple autoencoder anomaly detector for cardiovascular risk signals.
* Test on dataset; produce anomaly scoring and thresholds.
  **Deliverable:** `05_anomaly.ipynb` + anomaly scoring module.

## Week 16 — Anomaly endpoint & UI integration (POC)

* Add `/api/anomalies/` to list/store anomaly events for a user.
* Streamlit: anomalies dashboard (scores, suggested actions).
  **Deliverable:** anomaly endpoint + dashboard.

## Week 17 — Integration pass & code cleanup

* Ensure all endpoints share the same inference pipeline; deduplicate code.
* Add model_versioning entries and simple logging of predictions.
  **Deliverable:** integrated codebase and consolidated inference module.

## Week 18 — Testing & basic CI

* Add unit tests for key API endpoints and smoke tests for Streamlit pages.
* Set up basic GitHub Actions workflow (lint + tests).
  **Deliverable:** passing test suite and CI configured.

## Week 19 — Documentation & reproducibility

* Write `How to run` instructions, API usage examples, and model training README.
* Prepare notebooks and scripts required to reproduce main experiments.
  **Deliverable:** complete docs and reproducibility checklist.

## Week 20 — Report & presentation draft

* Draft thesis/report sections: Introduction, Data, Methods, Results, Discussion, Ethics.
* Prepare slides for defense/demo and a short demo script.
  **Deliverable:** draft report and slide deck.

## Week 21 — Final polishing & demo artifacts

* Final edits to code, docs, and notebooks; record a short demo video.
* Package deliverables: repo, Docker compose, model artifacts, report, slides, demo.
  **Deliverable:** final submission bundle and demo-ready Streamlit app.

---

## Priorities & scope guidance (quick)

* **Must-have (MVP):** DRF auth, multi-task prediction, SHAP explanations, clustering, Streamlit demo.
* **POC (if time):** recommendation uplift baseline and anomaly detector.
* **Optional advanced:** full causal mediation analysis or production-grade uplift evaluation - include as appendix/POC if time allows.

## Risk mitigation tips

* Start with **LightGBM / tabular models** (fast, explainable) before any NN experiments.
* Keep inference code single-source (shared by API and Streamlit).
* Treat causal/uplift as POC: document limitations and include results as exploratory.
* Commit small, frequent changes and keep notebooks reproducible.