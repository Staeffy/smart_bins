from pathlib import Path

class Config:
    RANDOM_SEED = 42
    ASSETS_PATH = Path('./assets')
    RAW_DATASET_FILE_PATH = ASSETS_PATH / 'raw_dataset' / 'Smart-Bins-Messwerte(1).xlsx'
    DATASET_PATH = ASSETS_PATH / 'data'
    CSV_DATASET_FILE_PATH = DATASET_PATH / 'smart_bins.csv'
    CLEANED_DATASET_FILE_PATH = DATASET_PATH / 'dataframe_cleaned.csv'
    USE_CASE_PREPARATION_DATASET_FILE_PATH = DATASET_PATH / 'use_case_dataframe.csv'
    FEATURES_PATH = ASSETS_PATH / 'features'
    TRAIN_FILE_PATH = FEATURES_PATH / 'train_X_Y.pt'
    TEST_FILE_PATH = FEATURES_PATH / 'test_X_Y.pt'
    PREDICT_FILE_PATH = FEATURES_PATH / 'predict_X_Y.pt'
    X_VARIABLE_FILE_PATH = FEATURES_PATH / 'x_variable.pt'


    MODELS_PATH = ASSETS_PATH / 'models'
    MODEL_NAME = 'lstm_model.pt'
    MODEL_FILE_PATH = MODELS_PATH / MODEL_NAME
    METRICS_FILE_PATH = ASSETS_PATH / 'metrics.json'

    # Data Preprocessing
    SEQ_LENGTH = 4

    # LSTM Settings
    NUMBER_EPOCHS = 2000
    LEARNING_RATE = 0.01
    INPUT_SIZE = 1
    HIDDEN_SIZE = 2
    NUMBER_OF_LAYERS = 1
    NUMBER_OF_CLASSES = 1
