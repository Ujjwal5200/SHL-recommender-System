import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name="shl_recommender"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)

    # File (rotating)
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler("logs/app.log", maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()  