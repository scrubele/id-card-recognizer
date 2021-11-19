import os
from pathlib import Path

from src.utils.logger import setup_logger

ABSOLUTE_FOLDER = os.path.join(os.path.abspath(os.path.join("SRC", os.pardir)))
PARENT_FOLDER = str(Path(__file__).parent)
DATA_FOLDER = os.path.join(ABSOLUTE_FOLDER, "data")
TEST_FOLDER = os.path.join(DATA_FOLDER, "test")
DEBUG_FOLDER = os.path.join(DATA_FOLDER, "debug")
TEMPLATES_FOLDER = os.path.join(DATA_FOLDER, "templates")
LOG_FOLDER = os.path.join(ABSOLUTE_FOLDER, "logs")
LOG_IMG_FOLDER = os.path.join(LOG_FOLDER, "images")

logger = setup_logger()
logger.LOG_FOLDER = LOG_FOLDER
logger.LOG_IMG_FOLDER = LOG_IMG_FOLDER
logger.info(f'Logging level: {logger.level}')
