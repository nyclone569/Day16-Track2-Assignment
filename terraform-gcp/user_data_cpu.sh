#!/bin/bash
set -e

# Log file
LOG_FILE="/var/log/ml-setup.log"
exec > >(tee -a $LOG_FILE) 2>&1

echo "=== Starting ML Environment Setup for LightGBM ==="
echo "Timestamp: $(date)"

# Update system
apt-get update -y
apt-get upgrade -y

# Install Python and dependencies
apt-get install -y python3 python3-pip python3-venv unzip wget

# Create working directory
mkdir -p /home/ubuntu/ml-benchmark
cd /home/ubuntu/ml-benchmark

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install ML packages
pip install --upgrade pip
pip install lightgbm scikit-learn pandas numpy flask kaggle

# Create benchmark script
cat > /home/ubuntu/ml-benchmark/benchmark.py << 'EOFPY'
import time
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

print("=== LightGBM Benchmark on GCP CPU Instance ===")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load dataset
print("\n[1/5] Loading Credit Card Fraud dataset...")
start_load = time.time()
df = pd.read_csv('/home/ubuntu/ml-benchmark/creditcard.csv')
load_time = time.time() - start_load
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Load time: {load_time:.2f} seconds")

# Prepare data
print("\n[2/5] Preparing train/test split...")
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Train model
print("\n[3/5] Training LightGBM model...")
start_train = time.time()
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    num_leaves=31,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc')
train_time = time.time() - start_train
print(f"Training completed in {train_time:.2f} seconds")

# Evaluate
print("\n[4/5] Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "auc_roc": roc_auc_score(y_test, y_pred_proba),
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred)
}

print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")

# Inference benchmark
print("\n[5/5] Benchmarking inference...")
single_start = time.time()
_ = model.predict(X_test.iloc[:1])
single_latency = (time.time() - single_start) * 1000  # ms

batch_start = time.time()
_ = model.predict(X_test.iloc[:1000])
batch_time = time.time() - batch_start
throughput = 1000 / batch_time

# Save results
results = {
    "instance_type": "n2-highmem-8 (GCP)",
    "dataset": "Credit Card Fraud Detection",
    "dataset_size": df.shape[0],
    "load_time_sec": round(load_time, 2),
    "train_time_sec": round(train_time, 2),
    "best_iteration": model.best_iteration_,
    "metrics": {k: round(v, 4) for k, v in metrics.items()},
    "inference_latency_ms": round(single_latency, 2),
    "inference_throughput_per_sec": round(throughput, 2)
}

with open('/home/ubuntu/ml-benchmark/benchmark_result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== Benchmark Complete ===")
print(json.dumps(results, indent=2))
EOFPY

# Set permissions
chown -R ubuntu:ubuntu /home/ubuntu/ml-benchmark

echo "=== Setup Complete ==="
echo "To run benchmark:"
echo "1. SSH into instance: gcloud compute ssh ai-cpu-node --zone=us-west1-b --tunnel-through-iap"
echo "2. Setup Kaggle credentials in ~/.kaggle/kaggle.json"
echo "3. Download dataset: kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p /home/ubuntu/ml-benchmark/"
echo "4. Run: cd /home/ubuntu/ml-benchmark && source venv/bin/activate && python3 benchmark.py"
