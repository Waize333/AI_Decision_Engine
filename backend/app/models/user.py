"""
backend/app/models/user.py
===========================
PURPOSE:
  SQLAlchemy ORM model for the 'users' table in PostgreSQL.
  Defines the schema (columns), relationships, and constraints.

WHAT THIS TABLE STORES:
  - User identity: email, hashed password
  - Role: admin | engineer | analyst | client
  - Status: is_active (soft delete instead of hard delete)

SOFT DELETES:
  We never actually DELETE rows from the users table.
  Instead we set is_active=False. This preserves audit history —
  if a user ran an inference in the past, we can still trace it.
"""

import enum

from sqlalchemy import Boolean, Enum, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class UserRole(str, enum.Enum):
    """
    Enumeration of all possible user roles.

    Using str + enum means values are stored as strings in the DB
    ("admin", "engineer", etc.) — readable and debuggable.

    Hierarchy (lowest → highest permission):
      client → analyst → engineer → admin
    """
    ADMIN = "admin"
    ENGINEER = "engineer"
    ANALYST = "analyst"
    CLIENT = "client"


class User(Base, UUIDMixin, TimestampMixin):
    """
    Represents a row in the 'users' PostgreSQL table.

    Columns:
      id           — UUID primary key (from UUIDMixin)
      email        — unique login identifier
      hashed_password  — bcrypt hash of the user's password
      full_name    — display name
      role         — RBAC role (controls permissions)
      is_active    — soft delete flag
      is_verified  — email verification status
      created_at   — from TimestampMixin
      updated_at   — from TimestampMixin

    Relationships:
      (future) → InferenceRequest (one user → many requests)
    """
    __tablename__ = "users"

    # Login credential — must be globally unique
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,       # fast lookup by email at login
        nullable=False,
    )

    # We NEVER store plain passwords — only bcrypt hashes
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # Optional display name (nullable is fine)
    full_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # RBAC role — drives all permission checks
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role_enum"),
        default=UserRole.CLIENT,        # new signups get least privilege
        nullable=False,
        index=True,                     # filter users by role in admin panel
    )

    # Soft delete: deactivated users can't log in but data is preserved
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # Email verification (can be extended with token later)
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User id={self.id!r} email={self.email!r} role={self.role!r}>"

    def has_role(self, required_role: UserRole) -> bool:
        """
        Check if this user's role meets or exceeds the required role.

        Role hierarchy (highest → lowest):
          admin > engineer > analyst > client

        Example:
          engineer_user.has_role(UserRole.ANALYST)  → True
          client_user.has_role(UserRole.ENGINEER)   → False
        """
        hierarchy = {
            UserRole.CLIENT: 0,
            UserRole.ANALYST: 1,
            UserRole.ENGINEER: 2,
            UserRole.ADMIN: 3,
        }
        return hierarchy.get(self.role, 0) >= hierarchy.get(required_role, 0)
