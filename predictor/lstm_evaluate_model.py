import json
import pandas as pd
from sklearn.metrics import mean_squared_error
from utils.helpers import read_tensors
from config import Config


def main_evaluate():
    test_X, test_Y = read_tensors(str(Config.TRAIN_FILE_PATH))
    print(type(test_X))


if __name__ == '__main__':
    main_evaluate()