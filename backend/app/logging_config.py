"""Structured logging configuration using structlog."""

import logging

import structlog


def setup_logging() -> None:
    """Configure structured logging for the application."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),  # switch to JSONRenderer() in production
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
