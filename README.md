# 🧠 AI Decision Engine Platform

> A production-grade AI inference platform with live drift detection, A/B testing, automated rollback, and feedback loops — built like OpenAI's backend.

---

## 🎯 What This Platform Does

| Feature | Description |
|---|---|
| **Live Inference** | Send data → get model predictions in real-time |
| **Drift Detection** | Automatically detects when model accuracy degrades |
| **A/B Testing** | Route traffic between model versions, measure winner |
| **Feedback Loop** | User feedback auto-generates training data |
| **Model Rollback** | One API call reverts to any prior working version |
| **RBAC Auth** | Role-based access (Admin, Engineer, Analyst, Client) |
| **Full Observability** | Prometheus + Grafana dashboards |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        CLIENT                           │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP/REST
┌─────────────────────────▼───────────────────────────────┐
│              API GATEWAY  (FastAPI)                      │
│         Rate Limiting · Auth · Request ID               │
└──────────┬──────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│            AUTH + RBAC MIDDLEWARE                        │
│         JWT Validation · Permission Guard               │
└──────────┬──────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│              INFERENCE ROUTER                           │
│   ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│   │  Model v1   │  │  Model v2   │  │  A/B Splitter │  │
│   └─────────────┘  └─────────────┘  └───────────────┘  │
└──────────┬──────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│           POST-PROCESSING & RESPONSE                    │
│      Confidence Score · Logging · Tracing               │
└──────────┬──────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│                  DATA LAYER                             │
│  PostgreSQL (metadata) · MongoDB (logs) · Redis (cache) │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AI_Decision_Engine/
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/v1/endpoints/ # Route handlers (auth, inference, etc.)
│   │   ├── core/             # Config, security, logging
│   │   ├── db/               # DB clients (postgres, mongo, redis)
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── services/         # Business logic layer
│   │   └── middleware/       # Auth, rate limiting, tracing
│   ├── alembic/              # Database migrations
│   └── tests/                # Unit + integration tests
├── docs/                     # Architecture docs & diagrams
├── scripts/                  # Dev/ops utility scripts
└── .github/workflows/        # CI/CD pipelines
```

---

## 🚀 Quick Start

```bash
# 1. Clone and enter
git clone <repo-url>
cd AI_Decision_Engine

# 2. Start all services
docker-compose up -d

# 3. Run database migrations
cd backend && alembic upgrade head

# 4. Start API server
uvicorn app.main:app --reload

# 5. Open docs
open http://localhost:8000/docs
```

---

## 📡 Core API Endpoints

| Method | Endpoint | Role Required | Description |
|---|---|---|---|
| POST | `/auth/register` | Public | Create new account |
| POST | `/auth/login` | Public | Get JWT token |
| POST | `/inference` | Client+ | Run model prediction |
| GET | `/inference/{id}` | Client+ | Get past result |
| POST | `/feedback` | Client+ | Submit feedback |
| GET | `/metrics` | Analyst+ | View model metrics |
| GET | `/metrics/drift` | Analyst+ | View drift alerts |
| POST | `/models/{id}/rollback` | Engineer+ | Rollback a model |

---

## 🔐 Roles & Permissions

| Role | Inference | Metrics | Deploy Model | Admin |
|---|---|---|---|---|
| Client | ✅ | ❌ | ❌ | ❌ |
| Analyst | ✅ | ✅ | ❌ | ❌ |
| Engineer | ✅ | ✅ | ✅ | ❌ |
| Admin | ✅ | ✅ | ✅ | ✅ |

---

## 🛠️ Tech Stack

- **FastAPI** — async Python web framework
- **PostgreSQL** — relational metadata store
- **MongoDB** — document store for inference logs
- **Redis** — caching + rate limiting
- **SQLAlchemy + Alembic** — ORM + migrations
- **Prometheus + Grafana** — metrics & dashboards
- **Docker + Docker Compose** — containerized deployment
- **GitHub Actions** — CI/CD pipeline

---

## 📊 Phases

| Phase | Topic | Status |
|---|---|---|
| 0 | System Design | 🔄 In Progress |
| 1 | Backend Foundation | 🔄 In Progress |
| 2 | AI Layer | ⏳ Planned |
| 3 | Database Design | ⏳ Planned |
| 4 | Feedback Loop | ⏳ Planned |
| 5 | Drift Detection | ⏳ Planned |
| 6 | A/B Testing | ⏳ Planned |
| 7 | Model Versioning & Rollback | ⏳ Planned |
| 8 | Monitoring & Metrics | ⏳ Planned |
| 9 | Deployment | ⏳ Planned |

---

> Built as a FAANG-level portfolio project. Every component is production-grade.
