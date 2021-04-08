from pathlib import Path

class Config:
    RANDOM_SEED = 42
    ASSETS_PATH = Path('./assets')
    RAW_DATASET_FILE_PATH = ASSETS_PATH / 'raw_dataset' / 'Smart-Bins-Messwerte(1).xlsx'
    DATASET_PATH = ASSETS_PATH / 'data'
    CSV_DATASET_FILE_PATH = DATASET_PATH / 'smart_bins.csv'
    FEATURES_PATH = ASSETS_PATH / 'features'
    MODELS_PATH = ASSETS_PATH / 'models'
    METRICS_FILE_PATH = ASSETS_PATH / 'metrics.json'