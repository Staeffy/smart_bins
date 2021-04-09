import torch
import logging
import warnings
import pandas as pd
import numpy as np
from config import Config
from utils.helpers import sliding_windows, save_train_test_splits 
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

def read_use_case_dataframe() -> pd.DataFrame:
    logger.info(str(Config.USE_CASE_PREPARATION_DATASET_FILE_PATH))
    use_case_dataframe = pd.read_csv(str(Config.USE_CASE_PREPARATION_DATASET_FILE_PATH), delimiter=",", index_col=[0])
    logger.info('Reading use case dataframe.')
    return use_case_dataframe

def train_test_split(use_case_dataframe: pd.DataFrame):

    sc = MinMaxScaler()
    training_set = use_case_dataframe.iloc[:,1:2].values
    training_data = sc.fit_transform(training_set)

    seq_length = int(Config.SEQ_LENGTH)
    x_variable, y_variable = sliding_windows(training_data, seq_length)

    train_size = int(len(y_variable) * 0.80)
    test_size = len(y_variable) - train_size

    dataX = Variable(torch.Tensor(np.array(x_variable)))
    dataY = Variable(torch.Tensor(np.array(y_variable)))

    trainX = Variable(torch.Tensor(np.array(x_variable[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y_variable[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x_variable[train_size:len(x_variable)])))
    testY = Variable(torch.Tensor(np.array(y_variable[train_size:len(y_variable)])))

    save_train_test_splits(dataX, dataY, trainX, trainY, testX, testY)
    logger.info('Train and test split is finsihed and saved.')

def main() -> None:
    use_case_dataframe = read_use_case_dataframe()
    train_test_split(use_case_dataframe)

if __name__ == '__main__':
    global logger
    logging.basicConfig(level = logging.DEBUG, filemode='a')
    file_handler = logging.FileHandler('log/create_features.log')
    logger = logging.getLogger('log_file')
    logger.addHandler(file_handler)
    main()
