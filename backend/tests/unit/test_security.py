"""
backend/tests/unit/test_security.py
=====================================
PURPOSE:
  Unit tests for core/security.py — password hashing and JWT logic.

  Unit tests test ONE function in isolation, with no DB or network.
  They are the fastest tests and catch the most bugs per second.

WHAT WE TEST:
  ✅ Password hashing produces a different string than the input
  ✅ Hashing the same password twice gives different hashes (bcrypt salt)
  ✅ verify_password correctly validates matching passwords
  ✅ verify_password rejects wrong passwords
  ✅ create_access_token produces a decodable token
  ✅ decode_token returns the correct subject
  ✅ Expired tokens raise JWTError
  ✅ Tampered tokens raise JWTError
"""

from datetime import timedelta

import pytest
from jose import JWTError

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    extract_user_id,
    hash_password,
    verify_password,
)


# ─── PASSWORD TESTS ───────────────────────────────────────────────────────────

class TestPasswordHashing:

    def test_hash_is_not_plaintext(self):
        """The hash must never equal the original password."""
        plain = "MySecretPass1"
        hashed = hash_password(plain)
        assert hashed != plain

    def test_hash_starts_with_bcrypt_prefix(self):
        """bcrypt hashes always start with '$2b$'."""
        hashed = hash_password("TestPass1")
        assert hashed.startswith("$2b$")

    def test_same_password_different_hashes(self):
        """
        Due to bcrypt's random salt, two hashes of the same
        password should always be different strings.
        This is intentional — prevents rainbow table attacks.
        """
        hashed1 = hash_password("SamePassword1")
        hashed2 = hash_password("SamePassword1")
        assert hashed1 != hashed2


class TestPasswordVerification:

    def test_correct_password_verifies(self):
        plain = "CorrectPass1"
        hashed = hash_password(plain)
        assert verify_password(plain, hashed) is True

    def test_wrong_password_fails(self):
        hashed = hash_password("CorrectPass1")
        assert verify_password("WrongPassword1", hashed) is False

    def test_empty_password_fails(self):
        hashed = hash_password("RealPass1")
        assert verify_password("", hashed) is False

    def test_case_sensitive(self):
        hashed = hash_password("Uppercase1")
        assert verify_password("uppercase1", hashed) is False


# ─── JWT TESTS ────────────────────────────────────────────────────────────────

class TestJWTTokens:

    def test_access_token_is_decodable(self):
        """Created tokens must be decodable with correct payload."""
        user_id = "abc-123-def"
        token = create_access_token(subject=user_id)
        payload = decode_token(token)

        assert payload["sub"] == user_id
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_token_includes_extra_claims(self):
        """Extra claims (like role) must appear in the decoded payload."""
        token = create_access_token(
            subject="user-123",
            extra_claims={"role": "admin", "email": "admin@test.com"}
        )
        payload = decode_token(token)
        assert payload["role"] == "admin"
        assert payload["email"] == "admin@test.com"

    def test_expired_token_raises_jwt_error(self):
        """Tokens with negative TTL should be immediately expired."""
        token = create_access_token(
            subject="user-123",
            expires_delta=timedelta(seconds=-1),   # already expired
        )
        with pytest.raises(JWTError):
            decode_token(token)

    def test_tampered_token_raises_jwt_error(self):
        """Modifying token payload invalidates the signature."""
        token = create_access_token(subject="user-123")
        # Flip one character in the signature section (last part)
        parts = token.split(".")
        parts[2] = parts[2][:-1] + ("X" if parts[2][-1] != "X" else "Y")
        tampered = ".".join(parts)

        with pytest.raises(JWTError):
            decode_token(tampered)

    def test_refresh_token_has_correct_type(self):
        """Refresh tokens must have type='refresh', not 'access'."""
        token = create_refresh_token(subject="user-456")
        payload = decode_token(token)
        assert payload["type"] == "refresh"
        assert payload["sub"] == "user-456"

    def test_extract_user_id_returns_subject(self):
        """extract_user_id is a convenience wrapper over decode_token."""
        token = create_access_token(subject="user-789")
        assert extract_user_id(token) == "user-789"

    def test_extract_user_id_returns_none_for_invalid_token(self):
        """Invalid token should return None (not raise)."""
        result = extract_user_id("totally.invalid.token")
        assert result is None
