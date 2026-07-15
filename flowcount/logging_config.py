"""Centralized logging configuration.

Provides a single :func:`setup_logging` entry point so the CLI, the web
dashboard, and the benchmark all configure logging the same way. Uses
``rich`` for pretty console output when available, falling back to the
standard library otherwise.
"""

import logging

_CONFIGURED = False


def setup_logging(level: int | str = logging.INFO, use_rich: bool = True) -> logging.Logger:
    """Configure root logging once and return the package logger.

    Args:
        level: Logging level as an int (``logging.INFO``) or name (``"INFO"``).
        use_rich: Use ``rich.logging.RichHandler`` for colorized output when
            the ``rich`` package is installed.

    Returns:
        The ``"objectdetection"`` logger.
    """
    global _CONFIGURED

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return logging.getLogger("objectdetection")

    handler: logging.Handler
    if use_rich:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, show_path=False)
            fmt, datefmt = "%(message)s", "[%X]"
        except ImportError:
            use_rich = False

    if not use_rich:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
        datefmt = "%H:%M:%S"

    logging.basicConfig(level=level, handlers=[handler], format=fmt, datefmt=datefmt)
    _CONFIGURED = True
    return logging.getLogger("objectdetection")
