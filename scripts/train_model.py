"""
scripts/train_model.py
========================
PURPOSE:
  Phase 2 — Train and serialize the fraud detection ML model.

  This script:
    1. Generates a synthetic training dataset (fraud detection)
    2. Preprocesses features (encoding, scaling)
    3. Trains a RandomForestClassifier
    4. Evaluates on a hold-out test set
    5. Saves the model as a .joblib artifact
    6. Registers the new version in PostgreSQL

WHY RANDOM FOREST FOR FRAUD DETECTION?
  - Handles mixed feature types (categorical + numerical)
  - Robust to outliers (credit amounts vary 10x)
  - Naturally produces class probabilities (needed for confidence score)
  - Easily explainable with feature importance
  - Battle-tested in production fraud systems (Stripe, PayPal use variants)

HOW TO RUN:
  cd backend
  python -m scripts.train_model --version v1 --samples 10000

DATA:
  In production, replace generate_synthetic_data() with a call to
  your actual historical transaction database.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── DATASET GENERATION ──────────────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic fraud detection dataset.

    In a real system this would be:
      SELECT * FROM transactions WHERE date > NOW() - INTERVAL '90 days'

    Features:
      transaction_amount   — dollars (0–10,000)
      merchant_category    — type of merchant (encoded)
      hour_of_day          — 0–23 (fraud peaks at night)
      is_international     — 0/1 (international transactions riskier)
      user_account_age_days — how old the account is (new accounts = higher risk)
      num_transactions_today — transaction velocity
      label                — 0=legitimate, 1=fraudulent

    Fraud rate: ~8% (realistic — most transactions are legitimate)
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * 0.08)
    n_legit = n_samples - n_fraud

    categories = ["grocery", "electronics", "restaurant", "travel", "gas", "online"]

    def make_legit(n):
        return pd.DataFrame({
            "transaction_amount": rng.lognormal(4.5, 1.2, n).clip(1, 5000),
            "merchant_category": rng.choice(categories, n, p=[0.35, 0.10, 0.25, 0.10, 0.12, 0.08]),
            "hour_of_day": rng.integers(7, 22, n),   # business hours
            "is_international": rng.choice([0, 1], n, p=[0.85, 0.15]),
            "user_account_age_days": rng.integers(180, 3000, n),
            "num_transactions_today": rng.integers(1, 8, n),
            "label": 0,
        })

    def make_fraud(n):
        return pd.DataFrame({
            "transaction_amount": rng.lognormal(6.5, 1.5, n).clip(200, 10000),  # higher amounts
            "merchant_category": rng.choice(categories, n, p=[0.05, 0.40, 0.05, 0.30, 0.05, 0.15]),
            "hour_of_day": rng.choice(list(range(0, 6)) + list(range(22, 24)), n),  # late night
            "is_international": rng.choice([0, 1], n, p=[0.30, 0.70]),  # more international
            "user_account_age_days": rng.integers(1, 90, n),   # new accounts
            "num_transactions_today": rng.integers(5, 20, n),  # high velocity
            "label": 1,
        })

    df = pd.concat([make_legit(n_legit), make_fraud(n_fraud)], ignore_index=True)
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Encode categorical features and return (X, y, feature_names).

    We use LabelEncoder for merchant_category. In production, use
    OneHotEncoder for better generalization to unseen categories.
    """
    le = LabelEncoder()
    df = df.copy()
    df["merchant_category_encoded"] = le.fit_transform(df["merchant_category"])

    feature_cols = [
        "transaction_amount",
        "merchant_category_encoded",
        "hour_of_day",
        "is_international",
        "user_account_age_days",
        "num_transactions_today",
    ]

    X = df[feature_cols].values
    y = df["label"].values
    return X, y, feature_cols


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(X_train, y_train, n_estimators: int = 200) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier.

    Key hyperparameters:
      n_estimators=200    — 200 trees (more = better, slower to train)
      max_depth=10        — prevents overfitting on synthetic data
      min_samples_leaf=5  — minimum samples per leaf (regularisation)
      class_weight="balanced" — compensates for 8% fraud → 92% legit imbalance
      n_jobs=-1           — use all CPU cores
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",      # critical for imbalanced fraud datasets
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ─── EVALUATION ──────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    """Compute all evaluation metrics and print a classification report."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"AUC-ROC: {metrics['auc_roc']}")

    return metrics


# ─── SAVE MODEL ──────────────────────────────────────────────────────────────

def save_model(model, feature_names: list, version_tag: str) -> tuple[Path, str]:
    """
    Serialize the trained model to disk as a .joblib file.

    Also computes a SHA256 hash of the training data parameters
    as a stand-in for a real data hash (which you'd compute from
    your actual dataset file).

    Returns: (artifact_path, data_hash)
    """
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    artifact_name = f"fraud_detector_{version_tag}.joblib"
    artifact_path = models_dir / artifact_name

    # Save both the model and the expected feature order
    payload = {
        "model": model,
        "feature_names": feature_names,
        "version_tag": version_tag,
    }
    joblib.dump(payload, artifact_path)

    # Hash the artifact as a reproducibility signature
    file_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    print(f"\n💾 Model saved: {artifact_path}")
    print(f"   SHA256: {file_hash[:16]}...")
    return artifact_path, file_hash


# ─── REGISTER IN DB ──────────────────────────────────────────────────────────

async def register_in_db(
    version_tag: str,
    artifact_path: Path,
    metrics: dict,
    feature_names: list,
    data_hash: str,
) -> None:
    """
    Upsert the trained model version into PostgreSQL.
    Sets it as the active version and deactivates the old one.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.core.config import settings
    from app.models.model_version import ModelStatus, ModelVersion

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with SessionLocal() as session:
        async with session.begin():
            # Deactivate current active version
            result = await session.execute(
                select(ModelVersion).where(ModelVersion.status == ModelStatus.ACTIVE)
            )
            for old in result.scalars().all():
                old.status = ModelStatus.DEPRECATED
                old.traffic_percentage = 0
                print(f"  ↓ Deprecated: {old.name} {old.version_tag}")

            # Create new version
            new_version = ModelVersion(
                name="fraud_detector",
                version_tag=version_tag,
                description=f"RandomForest fraud detector {version_tag}. AUC-ROC: {metrics['auc_roc']}",
                artifact_path=str(artifact_path),
                training_data_hash=data_hash,
                status=ModelStatus.ACTIVE,
                traffic_percentage=100,
                metrics=metrics,
                feature_schema={f: "float" for f in feature_names},
            )
            session.add(new_version)
            await session.flush()
            print(f"  ✅ Registered: fraud_detector {version_tag} (ID: {new_version.id})")

    await engine.dispose()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def main(version_tag: str, n_samples: int, register: bool):
    print(f"\n🧠 AI Decision Engine — Model Trainer")
    print(f"=" * 50)
    print(f"Version:  {version_tag}")
    print(f"Samples:  {n_samples:,}")

    # 1. Generate data
    print(f"\n[1/5] Generating {n_samples:,} synthetic transactions...")
    df = generate_synthetic_data(n_samples)
    fraud_rate = df["label"].mean()
    print(f"      Fraud rate: {fraud_rate:.1%}")

    # 2. Preprocess
    print("[2/5] Preprocessing features...")
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. Train
    print("[3/5] Training RandomForestClassifier...")
    model = train(X_train, y_train)
    print(f"      Trained with {model.n_estimators} trees")

    # 4. Evaluate
    print("[4/5] Evaluating on test set...")
    metrics = evaluate(model, X_test, y_test)

    # 5. Save
    print("[5/5] Saving model artifact...")
    artifact_path, data_hash = save_model(model, feature_names, version_tag)

    # Optional: register in DB
    if register:
        print("\n📋 Registering in database...")
        await register_in_db(version_tag, artifact_path, metrics, feature_names, data_hash)

    print(f"\n🎉 Done! Model {version_tag} ready for inference.")
    print(f"\n📊 Final Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fraud detection model")
    parser.add_argument("--version", default="v1", help="Version tag (e.g. v1, v2)")
    parser.add_argument("--samples", type=int, default=10_000, help="Training dataset size")
    parser.add_argument(
        "--register", action="store_true",
        help="Register the trained model in PostgreSQL after training"
    )
    args = parser.parse_args()
    asyncio.run(main(args.version, args.samples, args.register))
