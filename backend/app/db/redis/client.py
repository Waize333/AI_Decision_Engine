"""
backend/app/db/redis/client.py
================================
PURPOSE:
  Redis client for:
    1. Request caching — avoid hitting the model for identical inputs
    2. Rate limiting   — prevent API abuse (token bucket / sliding window)
    3. Hot config      — store active model version so inference router
                         can read it in <1ms without a DB call

WHY REDIS?
  Redis is an in-memory key-value store. Reads are ~0.1ms.
  It's perfect for:
    - Ephemeral data (cache, rate limits) — no need for persistence
    - High-frequency reads (every request checks rate limit)
    - Simple key-value lookups (active model config)

USAGE:
  from app.db.redis.client import get_redis, RedisClient

  async def check_rate_limit(user_id: str) -> bool:
      redis = await get_redis()
      return await RedisClient.is_rate_limited(redis, user_id)
"""

from typing import Optional

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── CLIENT ──────────────────────────────────────────────────────────────────

_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """
    Get or create the Redis connection pool.

    Returns a shared async Redis client.
    Raises ConnectionError if Redis is unreachable.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,       # return str instead of bytes
            max_connections=20,
            socket_connect_timeout=2,    # fail fast
            socket_timeout=2,
        )
        logger.info("redis_client_created", url=settings.REDIS_URL)
    return _redis_client


async def close_redis_connection() -> None:
    """Close Redis connection. Called on app shutdown."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("redis_connection_closed")


# ─── RATE LIMITING ───────────────────────────────────────────────────────────

class RateLimiter:
    """
    Sliding window rate limiter using Redis sorted sets.

    HOW IT WORKS:
      For each user, we store a Redis sorted set where:
        - Member = timestamp of a request
        - Score  = same timestamp

      On each request:
        1. Remove all members older than the window
        2. Count remaining members
        3. If count >= limit → reject (429 Too Many Requests)
        4. Else add current timestamp and allow

    This gives exact counts over a sliding time window, unlike
    fixed-window counters that can allow 2x the limit at boundaries.
    """

    @staticmethod
    async def is_rate_limited(
        redis: aioredis.Redis,
        identifier: str,              # usually user_id or IP address
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> bool:
        """
        Check if an identifier has exceeded the rate limit.

        Returns:
            True  → rate limit exceeded (block the request)
            False → under the limit (allow the request)
        """
        import time

        limit = limit or settings.RATE_LIMIT_REQUESTS
        window = window_seconds or settings.RATE_LIMIT_WINDOW_SECONDS

        key = f"rate_limit:{identifier}"
        now = time.time()
        window_start = now - window

        # Atomic pipeline — all commands execute together
        async with redis.pipeline(transaction=True) as pipe:
            # Remove timestamps outside the current window
            await pipe.zremrangebyscore(key, "-inf", window_start)
            # Count requests in current window
            await pipe.zcard(key)
            # Add current request
            await pipe.zadd(key, {str(now): now})
            # Set TTL so Redis auto-cleans idle keys
            await pipe.expire(key, window)

            results = await pipe.execute()

        request_count = results[1]  # result of zcard
        return request_count >= limit

    @staticmethod
    async def get_remaining(
        redis: aioredis.Redis,
        identifier: str,
    ) -> int:
        """
        How many requests can this identifier still make?
        Used to populate X-RateLimit-Remaining response header.
        """
        import time

        window = settings.RATE_LIMIT_WINDOW_SECONDS
        key = f"rate_limit:{identifier}"
        now = time.time()
        window_start = now - window

        await redis.zremrangebyscore(key, "-inf", window_start)
        current_count = await redis.zcard(key)
        return max(0, settings.RATE_LIMIT_REQUESTS - current_count)


# ─── CACHE ───────────────────────────────────────────────────────────────────

class CacheManager:
    """
    Simple key-value cache with TTL (time-to-live).

    Used to cache inference results for identical inputs so we don't
    call the model again for the same data.
    """

    INFERENCE_CACHE_TTL = 300       # 5 minutes
    MODEL_CONFIG_TTL = 60           # 1 minute (model config changes rarely)

    @staticmethod
    async def get(redis: aioredis.Redis, key: str) -> Optional[str]:
        """Get a cached value. Returns None on cache miss."""
        return await redis.get(key)

    @staticmethod
    async def set(
        redis: aioredis.Redis,
        key: str,
        value: str,
        ttl_seconds: int = INFERENCE_CACHE_TTL,
    ) -> None:
        """Set a value with a TTL. Redis auto-deletes after TTL expires."""
        await redis.setex(key, ttl_seconds, value)

    @staticmethod
    async def delete(redis: aioredis.Redis, key: str) -> None:
        """Invalidate a cached value."""
        await redis.delete(key)

    @staticmethod
    async def set_model_config(
        redis: aioredis.Redis,
        config: dict,
    ) -> None:
        """
        Store the active model routing config.
        Inference router reads this on every request.
        """
        import json
        await redis.setex(
            "model:active_config",
            CacheManager.MODEL_CONFIG_TTL,
            json.dumps(config),
        )

    @staticmethod
    async def get_model_config(
        redis: aioredis.Redis,
    ) -> Optional[dict]:
        """Read the active model config from cache."""
        import json
        raw = await redis.get("model:active_config")
        return json.loads(raw) if raw else None
