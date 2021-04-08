""" Helper functions """
import pandas as pd
import numpy as np
from typing import Tuple

def remove_duplicate(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe.drop_duplicates(keep="first", inplace=True)   
    return dataframe

def sliding_windows(data: pd.DataFrame, seq_length: int) -> Tuple[np.array,np.array]:
    x_variable = []
    y_variable = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x_variable.append(_x)
        y_variable.append(_y)

    return np.array(x_variable), np.array(y_variable)