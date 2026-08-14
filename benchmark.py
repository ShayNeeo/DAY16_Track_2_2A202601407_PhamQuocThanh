#!/usr/bin/env python3
"""
ML Benchmark Script: LightGBM Credit Card Fraud Detection
Measures: Data Loading Time, Training Time, Evaluation Metrics (AUC-ROC, F1, Precision, Recall, Accuracy),
Inference Latency (1 row), and Inference Throughput (1000 rows).
Outputs structured benchmark_result.json and logs a formatted summary table.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tabulate import tabulate


def find_dataset():
    candidates = [
        "creditcard.csv",
        "ml-benchmark/creditcard.csv",
        os.path.expanduser("~/ml-benchmark/creditcard.csv"),
        "/home/ubuntu/ml-benchmark/creditcard.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def run_benchmark(data_path=None, output_path="benchmark_result.json"):
    print("=" * 60)
    print("⚡ Starting LightGBM ML Benchmark on Credit Card Fraud Detection")
    print("=" * 60)

    if not data_path:
        data_path = find_dataset()
        if not data_path:
            print("❌ Dataset creditcard.csv not found!")
            print("Please download it with: kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/")
            sys.exit(1)

    print(f"📂 Dataset location: {data_path}")

    # 1. Measure Data Loading Time
    t0 = time.perf_counter()
    df = pd.read_csv(data_path)
    load_time = time.perf_counter() - t0
    print(f"✓ Data loaded in {load_time:.4f}s (Shape: {df.shape[0]} rows, {df.shape[1]} columns)")

    # Feature & target separation
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Train LGBMClassifier & Measure Training Time
    print("🚀 Training LGBMClassifier...")
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
    )
    training_time = time.perf_counter() - t0
    best_iteration = getattr(model, "best_iteration_", model.n_estimators) or model.n_estimators
    print(f"✓ Training finished in {training_time:.4f}s (Best iteration: {best_iteration})")

    # 3. Model Evaluation on Test Set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc = float(roc_auc_score(y_test, y_pred_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # 4. Measure Inference Latency (1 row) - Averaged over 100 runs
    sample_row = X_test.iloc[[0]]
    # Warmup
    for _ in range(10):
        _ = model.predict_proba(sample_row)

    n_latency_runs = 100
    t0 = time.perf_counter()
    for _ in range(n_latency_runs):
        _ = model.predict_proba(sample_row)
    latency_1row_ms = ((time.perf_counter() - t0) / n_latency_runs) * 1000.0

    # 5. Measure Inference Throughput (1000 rows)
    sample_1000 = X_test.iloc[:1000]
    # Warmup
    for _ in range(5):
        _ = model.predict_proba(sample_1000)

    n_throughput_runs = 10
    t0 = time.perf_counter()
    for _ in range(n_throughput_runs):
        _ = model.predict_proba(sample_1000)
    total_batch_time = time.perf_counter() - t0
    avg_1000_batch_time_ms = (total_batch_time / n_throughput_runs) * 1000.0
    throughput_rows_per_sec = (1000 * n_throughput_runs) / total_batch_time

    # Construct Results
    results = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "data_loading_time_seconds": round(load_time, 4),
        "training_time_seconds": round(training_time, 4),
        "best_iteration": int(best_iteration),
        "auc_roc": round(auc_roc, 6),
        "accuracy": round(accuracy, 6),
        "f1_score": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "inference_latency_1_row_ms": round(latency_1row_ms, 4),
        "inference_throughput_1000_rows_ms": round(avg_1000_batch_time_ms, 4),
        "inference_throughput_rows_per_sec": round(throughput_rows_per_sec, 2),
    }

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_path}")

    # Display Table
    table_data = [
        ["Thời gian load data", f"{load_time:.4f} s"],
        ["Thời gian training", f"{training_time:.4f} s"],
        ["Best iteration", str(best_iteration)],
        ["AUC-ROC", f"{auc_roc:.6f}"],
        ["Accuracy", f"{accuracy:.6f}"],
        ["F1-Score", f"{f1:.6f}"],
        ["Precision", f"{precision:.6f}"],
        ["Recall", f"{recall:.6f}"],
        ["Inference latency (1 row)", f"{latency_1row_ms:.4f} ms"],
        ["Inference throughput (1000 rows)", f"{avg_1000_batch_time_ms:.4f} ms ({throughput_rows_per_sec:.1f} rows/s)"],
    ]

    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY TABLE")
    print("=" * 60)
    print(tabulate(table_data, headers=["Metric", "Kết quả"], tablefmt="grid"))
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmark(dataset_arg)
