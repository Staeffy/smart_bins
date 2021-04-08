
import logging
import numpy as np
import pandas as pd
import torch
import warnings
from datetime import datetime

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from config import Config
from utils.helpers import remove_duplicate, sliding_windows

np.random.seed(Config.RANDOM_SEED)

def create_folder() -> None:
    Config.RAW_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

    logger.info('Creating %s', Config.DATASET_PATH)

def read_raw_data() -> pd.DataFrame:
    dataframe_raw = pd.read_excel(str(Config.RAW_DATASET_FILE_PATH))
    logger.info('Reading from %s', Config.RAW_DATASET_FILE_PATH)
    dataframe_raw.to_csv(str(Config.CSV_DATASET_FILE_PATH), index = None, header=True)
    logger.info('Saving to %s', Config.RAW_DATASET_FILE_PATH)
    dataframe_csv = pd.read_csv(str(Config.CSV_DATASET_FILE_PATH), delimiter=",")
    return dataframe_csv

  #  df_train, df_test = train_test_split(df, test_size=0.2, random_state=Config.RANDOM_SEED)
   # df_train.to_csv(str(Config.DATASET_PATH / 'train.csv'), index=None)
    #df_test.to_csv(str(Config.DATASET_PATH / 'test.csv'), index=None)

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
    
    return use_case_dataframe

def train_test_split(use_case_dataframe: pd.DataFrame):

    sc = MinMaxScaler()
    training_set = use_case_dataframe.iloc[:,1:2].values
    training_data = sc.fit_transform(training_set)

    seq_length = 4
    x_variable, y_variable = sliding_windows(training_data, seq_length)

    train_size = int(len(y_variable) * 0.80)
    test_size = len(y_variable) - train_size

    dataX = Variable(torch.Tensor(np.array(x_variable)))
    dataY = Variable(torch.Tensor(np.array(y_variable)))

    trainX = Variable(torch.Tensor(np.array(x_variable[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y_variable[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x_variable[train_size:len(x_variable)])))
    testY = Variable(torch.Tensor(np.array(y_variable[train_size:len(y_variable)])))
    print(testX)
    print(type(testX))

def main():
    create_folder()
    dataframe_csv = read_raw_data()
    dataframe_cleaned = data_cleaning(dataframe_csv)
    use_case_dataframe = use_case_preparation(dataframe_cleaned)
    train_test_split(use_case_dataframe)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level = logging.DEBUG)
    logger = logging.getLogger('log_file')
    main()

