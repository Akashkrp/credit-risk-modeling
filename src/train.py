"""Train and compare credit default (PD) models, tune the decision threshold, build risk bands,
benchmark inference latency, and write an audit-ready model card."""
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data_pipeline import RAW_CSV, ROOT, build_dataset, file_sha256

SEED = 42
CATEGORICAL = ["sex", "housing", "saving_accounts", "checking_account", "purpose"]
NUMERIC = ["age", "job", "credit_amount", "duration", "monthly_burden"]
FEATURES = CATEGORICAL + NUMERIC
RISK_BANDS = ["A (lowest)", "B", "C", "D", "E (highest)"]


def preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])


def candidates(pos_weight):
    return {
        "logistic_regression": (LogisticRegression(class_weight="balanced", max_iter=2000),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "random_forest": (RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
                          {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8],
                           "clf__min_samples_leaf": [1, 5]}),
        "extra_trees": (ExtraTreesClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
                        {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8],
                         "clf__min_samples_leaf": [1, 5]}),
        "xgboost": (XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", random_state=SEED, n_jobs=-1),
                    {"clf__n_estimators": [200, 400], "clf__max_depth": [2, 3, 4],
                     "clf__learning_rate": [0.03, 0.1], "clf__subsample": [0.8]}),
    }


def ks_statistic(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def main():
    df = build_dataset()
    X, y = df[FEATURES], df.is_default
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, (clf, grid) in candidates(pos_weight).items():
        search = GridSearchCV(Pipeline([("prep", preprocessor()), ("clf", clf)]), grid,
                              scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_tr, y_tr)
        p = search.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p)
        results[name] = {"cv_auc": round(search.best_score_, 4), "test_auc": round(auc, 4),
                         "test_gini": round(2 * auc - 1, 4), "test_ks": round(ks_statistic(y_te, p), 4),
                         "best_params": search.best_params_}
        fitted[name] = search.best_estimator_
        print(f"{name:20s} cv_auc={search.best_score_:.3f} test_auc={auc:.3f}")

    best_name = max(results, key=lambda n: results[n]["cv_auc"])
    model = fitted[best_name]
    p_te = model.predict_proba(X_te)[:, 1]

    # Lending cares more about catching defaulters than raw accuracy: choose the highest threshold
    # that still catches at least 80% of defaults on out-of-fold training predictions.
    oof = np.zeros(len(y_tr))
    for tr_idx, va_idx in cv.split(X_tr, y_tr):
        m = sklearn.base.clone(model).fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        oof[va_idx] = m.predict_proba(X_tr.iloc[va_idx])[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    threshold = max(t for t in thresholds if recall_score(y_tr, oof >= t) >= 0.80)
    y_hat_default, y_hat_tuned = p_te >= 0.5, p_te >= threshold

    # Risk bands: quintiles of training-set PD, applied to the test set
    edges = np.quantile(model.predict_proba(X_tr)[:, 1], [0.2, 0.4, 0.6, 0.8])
    bands = pd.cut(p_te, [-np.inf, *edges, np.inf], labels=RISK_BANDS)
    band_table = (pd.DataFrame({"band": bands, "default": y_te.values})
                  .groupby("band", observed=False).default.agg(["count", "mean"]).round(3))

    # Latency: single-application scoring (real-time decision). Thread-pool start-up dominates
    # one-row predictions, so benchmark the multi-threaded model against single-threaded inference.
    def single_row_latency(m):
        for _ in range(20):
            m.predict_proba(X_te.iloc[[0]])
        times = []
        for i in range(300):
            t0 = time.perf_counter()
            m.predict_proba(X_te.iloc[[i % len(X_te)]])
            times.append((time.perf_counter() - t0) * 1000)
        return np.percentile(times, 50), np.percentile(times, 95)

    before_p50, before_p95 = single_row_latency(model)
    if "n_jobs" in model.named_steps["clf"].get_params():
        model.named_steps["clf"].set_params(n_jobs=1)
    single_p50, single_p95 = single_row_latency(model)
    batch = pd.concat([X_te] * 50, ignore_index=True)
    t0 = time.perf_counter()
    model.predict_proba(batch)
    batch_rows_per_s = len(batch) / (time.perf_counter() - t0)

    metrics = {
        "best_model": best_name,
        "model_comparison": results,
        "threshold": {"default_0.5": {"recall_bad": round(recall_score(y_te, y_hat_default), 4),
                                      "precision_bad": round(precision_score(y_te, y_hat_default), 4)},
                      "tuned": {"value": round(float(threshold), 3),
                                "recall_bad": round(recall_score(y_te, y_hat_tuned), 4),
                                "precision_bad": round(precision_score(y_te, y_hat_tuned), 4)}},
        "risk_bands_test": band_table.reset_index().to_dict(orient="records"),
        "risk_band_edges": [round(float(e), 4) for e in edges],
        "latency": {"multithreaded_p50_ms": round(float(before_p50), 2),
                    "multithreaded_p95_ms": round(float(before_p95), 2),
                    "single_p50_ms": round(float(single_p50), 2),
                    "single_p95_ms": round(float(single_p95), 2),
                    "batch_rows_per_s": int(batch_rows_per_s)},
    }
    lineage = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": RAW_CSV.name, "source_sha256": file_sha256(RAW_CSV),
        "rows": len(df), "train_rows": len(X_tr), "test_rows": len(X_te), "seed": SEED,
        "features": FEATURES, "python": platform.python_version(),
        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
    }

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": float(threshold), "band_edges": edges, "features": FEATURES},
                ROOT / "models" / "credit_risk_model.joblib")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (ROOT / "reports" / "model_card.json").write_text(json.dumps({"lineage": lineage, **metrics}, indent=2, default=str))

    print(f"\nBest model: {best_name}")
    print(classification_report(y_te, y_hat_tuned, target_names=["good", "bad"]))
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
