"""
scripts/seed_db.py
===================
PURPOSE:
  Seeds the PostgreSQL database with the minimum data required to
  start using the platform:
    1. Admin user     — so you can log in and manage the system
    2. Model version  — so POST /inference has something to call

WHEN TO RUN:
  After running `alembic upgrade head` for the first time.
  Safe to re-run — uses "upsert" logic (insert if not exists).

HOW TO RUN:
  cd backend
  python -m scripts.seed_db

  Or with custom credentials:
  ADMIN_EMAIL=you@company.com ADMIN_PASSWORD=MyPass1 python -m scripts.seed_db

ENVIRONMENT:
  Reads DATABASE_URL from environment / .env file.
  Make sure your .env is configured before running this.
"""

import asyncio
import os
import sys

# Add backend directory to Python path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.security import hash_password
from app.models.model_version import ModelStatus, ModelVersion
from app.models.user import User, UserRole


# ─── SEED DATA ────────────────────────────────────────────────────────────────

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@aidecisionengine.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "AdminPass1")
ADMIN_NAME = os.getenv("ADMIN_NAME", "Platform Admin")

INITIAL_MODEL = {
    "name": "fraud_detector",
    "version_tag": "v1",
    "description": (
        "Initial fraud detection model. "
        "Binary classifier: 0=legitimate, 1=fraudulent."
    ),
    "artifact_path": "./models/fraud_detector_v1.joblib",
    "status": ModelStatus.ACTIVE,
    "traffic_percentage": 100,     # all traffic until A/B testing starts
    "metrics": {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.89,
        "f1": 0.90,
        "auc_roc": 0.96,
        "note": "Baseline trained model."
    },
    "feature_schema": {
        # Expected input features with their types
        # Used by the inference service for input validation hints
        "transaction_amount": "float",
        "merchant_category": "string",
        "hour_of_day": "int",
        "is_international": "bool",
        "user_account_age_days": "int",
        "num_transactions_today": "int",
    },
}


# ─── SEED FUNCTIONS ───────────────────────────────────────────────────────────

async def seed_admin(session) -> None:
    """Create the admin user if it doesn't exist."""
    result = await session.execute(
        select(User).where(User.email == ADMIN_EMAIL)
    )
    existing = result.scalar_one_or_none()

    if existing:
        print(f"  ✓ Admin user already exists: {ADMIN_EMAIL}")
        return

    admin = User(
        email=ADMIN_EMAIL,
        hashed_password=hash_password(ADMIN_PASSWORD),
        full_name=ADMIN_NAME,
        role=UserRole.ADMIN,
        is_active=True,
        is_verified=True,    # admin is pre-verified
    )
    session.add(admin)
    await session.flush()
    print(f"  ✅ Created admin user: {ADMIN_EMAIL}")
    print(f"     Password: {ADMIN_PASSWORD}  ← Change this immediately!")
    print(f"     User ID:  {admin.id}")


async def seed_model_version(session) -> None:
    """Create the initial model version if it doesn't exist."""
    result = await session.execute(
        select(ModelVersion).where(
            ModelVersion.name == INITIAL_MODEL["name"],
            ModelVersion.version_tag == INITIAL_MODEL["version_tag"],
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        print(f"  ✓ Model version already exists: {INITIAL_MODEL['name']} {INITIAL_MODEL['version_tag']}")
        return

    model = ModelVersion(
        name=INITIAL_MODEL["name"],
        version_tag=INITIAL_MODEL["version_tag"],
        description=INITIAL_MODEL["description"],
        artifact_path=INITIAL_MODEL["artifact_path"],
        status=INITIAL_MODEL["status"],
        traffic_percentage=INITIAL_MODEL["traffic_percentage"],
        metrics=INITIAL_MODEL["metrics"],
        feature_schema=INITIAL_MODEL["feature_schema"],
    )
    session.add(model)
    await session.flush()
    print(f"  ✅ Created model version: {model.name} {model.version_tag}")
    print(f"     Status: {model.status.value} ({model.traffic_percentage}% traffic)")
    print(f"     ID: {model.id}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n🌱 AI Decision Engine — Database Seeder")
    print("=" * 50)
    print(f"Database: {settings.DATABASE_URL}")
    print()

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with SessionLocal() as session:
        try:
            print("Seeding users...")
            await seed_admin(session)

            print("\nSeeding model versions...")
            await seed_model_version(session)

            await session.commit()
            print("\n✅ Seeding complete!")

        except Exception as e:
            await session.rollback()
            print(f"\n❌ Seeding failed: {e}")
            raise

    await engine.dispose()
    print("\n📋 Next steps:")
    print(f"  1. Login: POST /api/v1/auth/login")
    print(f"     Body: {{\"email\": \"{ADMIN_EMAIL}\", \"password\": \"{ADMIN_PASSWORD}\"}}")
    print(f"  2. Change the admin password immediately in production!")
    print(f"  3. Run POST /api/v1/inference to test the model")


if __name__ == "__main__":
    asyncio.run(main())
