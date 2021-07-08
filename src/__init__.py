import os
from pathlib import Path

ABSOLUTE_FOLDER = os.path.join(os.path.abspath(os.path.join("SRC", os.pardir)))
PARENT_FOLDER = str(Path(__file__).parent)
DATA_FOLDER = os.path.join(ABSOLUTE_FOLDER, "data")
TEST_FOLDER = os.path.join(DATA_FOLDER, "test")
DEBUG_FOLDER = os.path.join(DATA_FOLDER, "debug")
TEMPLATES_FOLDER = os.path.join(DATA_FOLDER, "templates")
