"""
Production authentication and authorization.
Supports JWT tokens, API keys, and OAuth2.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Security Schemes ───────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Models ─────────────────────────────────────────────────

class TokenPayload(BaseModel):
    sub: str
    exp: int
    iat: int
    jti: str
    role: str = "user"
    scopes: list[str] = []


class User(BaseModel):
    user_id: str
    email: str
    role: str = "user"
    is_active: bool = True
    scopes: list[str] = []


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


# ── In-Memory Stores (Replace with DB in production) ──────

_api_keys: dict[str, User] = {
    "dev-api-key-12345": User(
        user_id="dev-user",
        email="dev@psrit.com",
        role="admin",
        scopes=["read", "write", "admin"],
    ),
}

_users: dict[str, dict] = {
    "admin@psrit.com": {
        "user_id": "admin-001",
        "email": "admin@psrit.com",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "scopes": ["read", "write", "admin"],
    },
}

_revoked_tokens: set[str] = set()


# ── Token Operations ──────────────────────────────────────

def create_access_token(
    user: User,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.jwt_expiration_minutes))

    payload = {
        "sub": user.user_id,
        "email": user.email,
        "role": user.role,
        "scopes": user.scopes,
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": str(uuid4()),
    }

    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    logger.info("token_created", user=user.user_id, expires=str(expire))
    return token


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        token_data = TokenPayload(**payload)

        # Check revocation
        if token_data.jti in _revoked_tokens:
            raise HTTPException(status_code=401, detail="Token has been revoked")

        return token_data
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def revoke_token(jti: str) -> None:
    """Revoke a token by its JTI."""
    _revoked_tokens.add(jti)
    logger.info("token_revoked", jti=jti)


def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    user_data = _users.get(email)
    if not user_data:
        return None
    if not pwd_context.verify(password, user_data["hashed_password"]):
        return None
    return User(
        user_id=user_data["user_id"],
        email=user_data["email"],
        role=user_data["role"],
        scopes=user_data.get("scopes", []),
    )


# ── Dependency Injectors ──────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_scheme),
) -> User:
    """
    Authenticate via JWT bearer token OR API key.
    Used as a FastAPI dependency.
    """
    # Try API key first
    if api_key and api_key in _api_keys:
        user = _api_keys[api_key]
        if user.is_active:
            return user

    # Try JWT bearer
    if credentials:
        token_data = decode_token(credentials.credentials)
        return User(
            user_id=token_data.sub,
            email="",
            role=token_data.role,
            scopes=token_data.scopes,
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Provide a Bearer token or API key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_scope(required_scope: str):
    """Dependency factory for scope-based authorization."""
    async def _check_scope(user: User = Depends(get_current_user)) -> User:
        if required_scope not in user.scopes and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {required_scope}",
            )
        return user
    return _check_scope


class RateLimiter:
    """Simple in-memory rate limiter per user."""

    def __init__(self, rpm: int = 60):
        self.rpm = rpm
        self._requests: dict[str, list[float]] = {}

    def check(self, user_id: str) -> bool:
        now = time.time()
        window_start = now - 60

        if user_id not in self._requests:
            self._requests[user_id] = []

        # Clean old entries
        self._requests[user_id] = [
            t for t in self._requests[user_id] if t > window_start
        ]

        if len(self._requests[user_id]) >= self.rpm:
            return False

        self._requests[user_id].append(now)
        return True


rate_limiter = RateLimiter()
