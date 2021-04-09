
import logging
import numpy as np
import pandas as pd
import torch
import warnings
from datetime import datetime
from config import Config
from utils.helpers import remove_duplicate, sliding_windows, save_train_test_splits

np.random.seed(Config.RANDOM_SEED)

def create_folder() -> None:
    Config.RAW_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)
    Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)
    Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

    logger.info('Creating of folders is finished.')

def read_raw_data() -> pd.DataFrame:
    dataframe_raw = pd.read_excel(str(Config.RAW_DATASET_FILE_PATH))
    logger.info('Reading from %s', Config.RAW_DATASET_FILE_PATH)
    dataframe_raw.to_csv(str(Config.CSV_DATASET_FILE_PATH), index = None, header=True)
    logger.info('Saving to %s', Config.RAW_DATASET_FILE_PATH)
    dataframe_csv = pd.read_csv(str(Config.CSV_DATASET_FILE_PATH), delimiter=",")
    return dataframe_csv

def data_cleaning(dataframe: pd.DataFrame) -> pd.DataFrame:

    dataframe_cleaned = remove_duplicate(dataframe)

    dataframe_cleaned.rename(columns={"BauartID":"bauart_id", 
                    "GenMesswertID":"gen_messwert_id",
                    "Messzeitpunkt":"messzeitpunkt",
                    "Messwert":"messwert",
                    "Messwerttyp":"messwerttyp",
                    "Füllstand":"fuellstand",
                    "SensortypID":"sensortyp_id",
                    "SensorID":"sensor_id",
                    "Behältertyp": "behaeltertyp",
                    "Innenhöhe":"innenhoehe",
                    "Abfallfraktion":"abfallfraktion",
                    "Standort":"standort"}, inplace=True)

    dataframe_cleaned['messzeitpunkt']= pd.to_datetime(dataframe_cleaned['messzeitpunkt'])
    dataframe_cleaned['datum'] = dataframe_cleaned['messzeitpunkt'].dt.strftime('%d/%m/%Y')
    dataframe_cleaned['uhrzeit'] = dataframe_cleaned['messzeitpunkt'].dt.strftime('%H:%M')
    dataframe_cleaned['wochentag'] = dataframe_cleaned['messzeitpunkt'].apply(pd.Timestamp.weekday)
    dataframe_cleaned.to_csv(str(Config.CLEANED_DATASET_FILE_PATH))
    logger.info('Data cleaning stage is finished.')
    
    return dataframe_cleaned

def use_case_preparation(dataframe_cleaned: pd.DataFrame) -> pd.DataFrame:
    location = dataframe_cleaned["standort"]=="Schifferstr. 220"
    sensor_type = dataframe_cleaned["sensortyp_id"]=="MOBA_FLS"
    waste_type = dataframe_cleaned["abfallfraktion"]=="WG"
    real_check = dataframe_cleaned["sensortyp_id"]=="BEOBACHTUNG"
    filtered_data = dataframe_cleaned[(location) & (sensor_type) & (waste_type)]
    filtered_real = dataframe_cleaned[(location) & (real_check) & (waste_type)]
    filtered_data.reset_index(drop=True, inplace=True)

    use_case_dataframe = filtered_data[['messzeitpunkt','messwert']]
    use_case_dataframe['messwert'] = use_case_dataframe.messwert.astype(int)
    use_case_dataframe.to_csv(str(Config.USE_CASE_PREPARATION_DATASET_FILE_PATH))
    logger.info('Use case preparation is finished.')
    return use_case_dataframe

def main() -> None:
    create_folder()
    dataframe_csv = read_raw_data()
    dataframe_cleaned = data_cleaning(dataframe_csv)
    use_case_preparation(dataframe_cleaned)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level = logging.DEBUG, filemode='a')
    file_handler = logging.FileHandler('log/create_dataset.log')
    logger = logging.getLogger('log_file')
    logger.addHandler(file_handler)
    main()
