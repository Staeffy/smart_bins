""" Creates folder structure for assets."""
import logging
from config import Config


def create_folder() -> None:
    Config.RAW_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)
    Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)
    Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    Config.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    Config.PLOT_PATH.mkdir(parents=True, exist_ok=True)
    Config.METRICS_PATH.mkdir(parents=True, exist_ok=True)

    logger.info('Creating of folders is finished.')

if __name__ == '__main__':
    global logger
    logging.basicConfig(level = logging.DEBUG, filemode='a')
    file_handler = logging.FileHandler('log/create_folders.log')
    logger = logging.getLogger('create_folders')
    logger.addHandler(file_handler)
    create_folder()