import os

from src.config import ARTIFACT_DIR


def create_directories():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)