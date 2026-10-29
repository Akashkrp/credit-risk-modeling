# Credit Risk Modeling

An end-to-end probability-of-default (PD) model built on the
[German Credit dataset](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)
(1,000 loan applications, 30% defaults). It covers the SQL data pipeline, EDA, model comparison,
handling class imbalance, tuning the decision threshold, risk-band segmentation, latency benchmarking
and an interactive Streamlit app for scoring applications.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Data pipeline | `src/data_pipeline.py` | Loads the raw CSV into SQLite and builds a clean table with SQL: imputes missing account data as `unknown`, derives `monthly_burden` and `age_band`, and encodes the target |
| EDA | `src/eda.py` | Default rates by segment, chi-square and Mann-Whitney U tests, charts written to `reports/` |
| Modelling | `src/train.py` | Logistic Regression, Random Forest, Extra Trees and XGBoost, tuned with GridSearchCV (5-fold stratified, ROC-AUC). Class imbalance is handled with `class_weight` / `scale_pos_weight` |
| Policy threshold | `src/train.py` | Picks the decision threshold from out-of-fold predictions so that at least 80% of defaulters are caught |
| Risk bands | `src/train.py` | Groups applicants into PD quintiles A–E |
| Latency | `src/train.py` | p50/p95 single-application scoring time and batch throughput |
| Audit trail | `reports/model_card.json` | Source-file SHA-256, row counts, seed, features, library versions and all metrics |
| App | `app.py` | Streamlit form that returns the PD, the risk band and an early-warning flag |

## Results (hold-out test set, 200 applications)

| Model | CV ROC-AUC | Test ROC-AUC | Gini | KS |
|---|---|---|---|---|
| Logistic Regression | 0.751 | 0.752 | 0.50 | 0.48 |
| **Random Forest (selected)** | **0.771** | **0.788** | **0.58** | **0.48** |
| Extra Trees | 0.752 | 0.774 | 0.55 | 0.47 |
| XGBoost | 0.764 | 0.776 | 0.55 | 0.44 |

- **Threshold tuning:** defaulter recall rose from 68% (threshold 0.5) to **85%** (tuned threshold 0.405)
- **Risk bands:** default rate goes from 8.6% in band A to 67.5% in band E, a 7.8× separation
- **Latency:** single-threaded inference cut p50 scoring time from 40.7 ms to **10.6 ms** (−74%)
- **Strongest drivers (EDA):** checking-account balance (χ² = 123.7), savings, loan duration, age < 25

## Run

```bash
pip install -r requirements.txt
cd src
python eda.py
python train.py
cd ..
streamlit run app.py
```

Inspired by Onur Baltaci's
[Credit Risk Modeling tutorial](https://youtu.be/lhI999ETwmA).
