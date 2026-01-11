"""Logging configuration utilities."""

from __future__ import annotations

import logging
from typing import Dict

from voiceassistant.config import LoggingConfig


def configure_logging(config: LoggingConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for module, level in config.module_levels.items():
        logging.getLogger(module).setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
