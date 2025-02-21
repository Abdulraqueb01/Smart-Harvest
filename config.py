import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

os.makedirs(BASE_DIR / 'models', exist_ok=True)
os.makedirs(BASE_DIR / 'data', exist_ok=True)

MODEL_PATH = BASE_DIR / 'models' / 'Model.pkl'
LABELS_PATH = BASE_DIR / 'models' / 'labels.pkl'
STATES_SEASON_PATH = BASE_DIR / 'data' / 'StatesSeason.json'
FERTILIZER_CSV_PATH = BASE_DIR / 'data' / 'fertilizer.csv'

def validate_paths():
    """Validate that all required files exist"""
    missing_files = []
    for path in [MODEL_PATH, LABELS_PATH, STATES_SEASON_PATH, FERTILIZER_CSV_PATH]:
        if not path.exists():
            missing_files.append(str(path))
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
