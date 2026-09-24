"""Central logging configuration for the project."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a single stream handler.

    Args:
        level: Logging level (e.g. logging.INFO or logging.DEBUG).
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured; avoid stacking duplicate handlers.
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)
