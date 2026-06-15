import logging


def enable_debug_file(filename="package_debug.txt"):
    """Configures the package logger to output to a file."""
    logger = logging.getLogger("quicksimsbio")  # Target the top-level namespace
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers if called twice
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(filename, mode="w")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.debug("File logging initialized for quicksimsbio")
