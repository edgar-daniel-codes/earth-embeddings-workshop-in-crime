### Summer Internship - Earth Embeddings
### Utils - Production helpers (logging)
### By Edgar Daniel


"""

Production helpers shared by every pipeline script 
used by all ``__main__`` showcase entry points.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_LOG_PATH = "./logs/etl.log"


### -------------------------------------------------------------------------------
### Functions ---------------------------------------------------------------------


def init_logger(
    log_path: str | Path | None = DEFAULT_LOG_PATH,
    name: str = "earth_embeddings_etl",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or reset) the file logger used by the pipeline scripts.

    Parameters
    ----------
    log_path : str | Path | None
        Log file location; parent directories are created. ``None`` falls
        back to :data:`DEFAULT_LOG_PATH`.
    name : str
        Logger name; a fixed name makes repeated calls reuse the logger.
    level : int
        Logging level applied to the logger and its file handler.
    """
    log_path = Path(log_path) if log_path is not None else Path(DEFAULT_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers so re-initialisation never duplicates output.
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", delay=False)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    )

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
