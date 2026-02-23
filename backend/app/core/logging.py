"""
backend/app/core/logging.py
===========================
PURPOSE:
  Configure structured JSON logging for the entire application
  using the `structlog` library.

WHY STRUCTURED/JSON LOGGING?
  Plain text logs: "User 123 logged in at 10:45"
  JSON logs:       {"event": "user_login", "user_id": "123", "timestamp": "...", "level": "info"}

  JSON logs are machine-parseable — Grafana, Datadog, and
  Elasticsearch can filter/aggregate them without regex.
  This is the industry standard for production services.

USAGE:
  from app.core.logging import get_logger
  logger = get_logger(__name__)
  logger.info("inference_complete", request_id="abc", latency_ms=42)
"""

import logging
import sys
from typing import Optional

import structlog
from structlog.types import EventDict, WrappedLogger

from app.core.config import settings


def add_app_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Structlog processor: inject global context into every log line.

    Processors are functions that transform the event dict before
    rendering. This one adds the app name and environment so every
    log line carries context about WHERE it came from.
    """
    event_dict["app"] = settings.APP_NAME
    event_dict["env"] = settings.APP_ENV
    return event_dict


def configure_logging() -> None:
    """
    Set up structlog and wire it into Python's standard logging.

    Called once at application startup (in main.py).
    After this, all loggers — including third-party libraries
    that use standard logging — emit structured JSON.
    """

    # Configure the standard library logging level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",    # structlog handles the full format
        stream=sys.stdout,
        level=log_level,
    )

    # Choose renderer based on environment
    if settings.LOG_FORMAT == "json" or settings.is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty colored output for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            # Add log level (info, warning, error)
            structlog.stdlib.add_log_level,
            # Add timestamp in ISO-8601 format
            structlog.processors.TimeStamper(fmt="iso"),
            # Add source file + line number (great for debugging)
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            # Our custom processor — adds app/env
            add_app_context,
            # Format exceptions as structured data
            structlog.processors.format_exc_info,
            # Final render step
            renderer,
        ],
        # Use standard library logger as the backend
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a named logger instance.

    Args:
        name: Usually __name__ of the calling module.
              Appears in logs as {"logger": "app.services.inference", ...}

    Example:
        logger = get_logger(__name__)
        logger.info("model_loaded", version="v2", path="/models/v2.pkl")
        logger.error("db_connection_failed", error=str(e), retry=3)
    """
    return structlog.get_logger(name)
