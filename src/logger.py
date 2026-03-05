import logging
import os
import sys
import io
from logging.handlers import RotatingFileHandler

def setup_logger(name="shl_recommender"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler with fallback for different environments (Jupyter, etc.)
    try:
        # Try to use sys.stdout.buffer for proper encoding (standard Python)
        console = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'))
    except AttributeError:
        # Fallback for Jupyter notebooks and other environments where sys.stdout.buffer doesn't exist
        console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)

    # File (rotating) - with UTF-8 encoding
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler("logs/app.log", maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
