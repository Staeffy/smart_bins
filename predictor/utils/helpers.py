""" Helper functions """
import torch
import pandas as pd
import numpy as np
from typing import Tuple
from config import Config

def read_tensors(path_tensors: str) -> Tuple[torch.Tensor, torch.Tensor]:

    loaded = torch.load(path_tensors)
    tensor_X = loaded[list(loaded.keys())[0]]
    tensor_Y = loaded[list(loaded.keys())[1]]

    return tensor_X, tensor_Y

def remove_duplicate(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Function for removing duplicates from dataframe.

    Args:
        dataframe (pd.DataFrame): A given dataframe for removing duplicates.

    Returns:
        pd.DataFrame: Dataframe without any duplicates.
    """
    dataframe.drop_duplicates(keep="first", inplace=True)   
    return dataframe

def save_train_test_splits( predictX: torch.Tensor,
                            predictY: torch.Tensor,
                            trainX: torch.Tensor,
                            trainY: torch.Tensor,
                            testX: torch.Tensor,
                            testY: torch.Tensor) -> None:

    names_predict = {'predict_X': predictX, 'predict_Y': predictY}
    names_train = {'train_X': trainX, 'train_Y': trainY}
    names_test = {'test_X': testX, 'test_Y': testY}


    torch.save(names_predict, str(Config.PREDICT_FILE_PATH))
    torch.save(names_train, str(Config.TRAIN_FILE_PATH))
    torch.save(names_test, str(Config.TEST_FILE_PATH))



def sliding_windows(data: pd.DataFrame, seq_length: int) -> Tuple[np.array,np.array]:
    """Function for creating a sliding window for the time series data.

    Args:
        data (pd.DataFrame): [description]
        seq_length (int): [description]

    Returns:
        Tuple[np.array,np.array]: [description]
    """
    x_variable = []
    y_variable = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x_variable.append(_x)
        y_variable.append(_y)

    return np.array(x_variable), np.array(y_variable)