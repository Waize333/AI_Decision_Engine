# AI Decision Engine 

A production-grade, asynchronous AI inference platform built with FastAPI, PostgreSQL, MongoDB, and Redis. It acts as a comprehensive MLOps backend, providing live inference routing, A/B testing, automatic drift detection, instantaneous rollbacks, and data flywheels (feedback loops). 

Built to replicate the architecture used by top-tier tech companies for deploying machine learning models at scale.

##  Features

- **Live Inference API:** Ultra-low latency endpoint for serving model predictions.
- **A/B Testing (Canary Rollouts):** Safely route traffic (e.g., 90/10 split) between stable and experimental models.
- **Instant Rollbacks:** Demote broken models and instantly route 100% of traffic to a stable baseline via a single API call.
- **Drift Detection:** Background analytics automatically detect feature/confidence drift and alert your team.
- **Feedback Loops:** Ingest ground-truth feedback from clients to automatically generate future training datasets.
- **Role-Based Access Control (RBAC):** JWT authentication with distinct roles (Admin, Engineer, Analyst, Client).
- **Hybrid Polyglot Database:** 
  - *PostgreSQL* for strict relational state (Users, Metadata).
  - *MongoDB* for highly flexible, high-volume logs (Inference records, Feedback).
  - *Redis* for caching and rate-limiting.

##  Architecture

The engine is built around a non-blocking asynchronous router. When a request hits the gateway, it is authenticated, rate-limited, and cached. The actual model inference happens in memory, and the results are returned to the client *before* the transaction is logged asynchronously to MongoDB.

##  Tech Stack

- **Framework:** FastAPI (Python)
- **Databases:** PostgreSQL (asyncpg), MongoDB (Motor), Redis (aioredis)
- **Machine Learning:** scikit-learn, joblib, pandas, numpy
- **Infrastructure:** Docker, Docker Compose
- **Monitoring:** Prometheus, Grafana

##  Installation & Setup

### 1. Prerequisites
- [Docker](https://www.docker.com/) and Docker Compose
- Python 3.10+

### 2. Start the Databases
The platform requires PostgreSQL, MongoDB, and Redis to be running.
```bash
docker-compose up -d
```

### 3. Setup Python Environment
Create a virtual environment and install the dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 4. Initialize Database & Train Baseline Model
Set up the database schema using Alembic, train the initial Random Forest model, and seed the initial Admin user:
```bash
# Run database migrations
cd backend
alembic upgrade head
cd ..

# Train the baseline AI model and save the artifact
python -m scripts.train_model --version v1 --samples 10000 

# Seed the database (Creates Admin user and registers the v1 model)
python -m scripts.seed_db
```
*(Note: The `seed_db` script will print out your Admin login credentials. Keep these safe!)*

### 5. Start the Engine
Start the FastAPI application:
```bash
cd backend
uvicorn app.main:app --reload
```
Navigate to **`http://localhost:8000/docs`** to access the interactive Swagger API documentation.

## 📖 Usage Guide

1. **Authentication:** Log in via `POST /api/v1/auth/login` using your Admin credentials. Copy the JWT token and use it to authorize subsequent requests.
2. **Inference:** Test the AI via `POST /api/v1/inference`. Submit sample features to get a prediction, confidence score, and unique request ID.
3. **Submit Feedback:** Send ground-truth feedback via `POST /api/v1/feedback` using the `request_id` to build a dataset for your next model.
4. **Drift Detection:** Run `python -m scripts.run_drift_detection` to aggregate recent inferences and calculate drift scores.
5. **A/B Testing:** Train a new model (`python -m scripts.train_model --version v2`), register an experiment (`POST /api/v1/experiments`), and dynamically route traffic to test its performance safely.

##  License

**Non-Commercial License (Custom)**

Copyright (c) 2026. All rights reserved.

This project and its source code are provided for educational, portfolio, and personal use only. **You may not use this software for any commercial purposes, nor distribute it commercially, without explicit written permission from the author.** 

For commercial licensing inquiries, please contact the repository owner.
