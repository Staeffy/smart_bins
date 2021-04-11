""" LSTM Evaluator."""
import logging
import json
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from utils.helpers import read_tensors, r2_score
from config import Config
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

def calculate_metrics(model: object, test_X: torch.tensor, test_Y: torch.tensor):
    """Function to calcuate the model metrics.

    Args:
        model (object): Model to calcuate the metrics.
        float ([torch.Tensor]): Torch tensor with the test data.
        float ([torch.Tensor]): Torch tensor with the target test data.

    Returns:
        Tuple[float, float, float]: Mse, rmse and r2 values. 
    """
    model.eval()
    criterion = torch.nn.MSELoss()
    outputs = model(test_X)

    mse_tensor = criterion(outputs, test_Y)
    mse = mse_tensor.item() 

    rmse_tensor = torch.sqrt(criterion(outputs, test_Y))
    rmse = rmse_tensor.item() 
    
    r2_tensor = r2_score(test_Y, outputs)
    r2 = r2_tensor.item() 
    
    return mse, rmse, r2

def perform_save_metrics(model: object) -> None:
    """Function to load the model and performing the calculation of metrics.

    Args:
        model (object): Model for performing the calculations.
    """
    test_X, test_Y = read_tensors(str(Config.TEST_FILE_PATH))
    mse_test, rmse_test, r2_test = calculate_metrics(model, test_X, test_Y)

    train_X, train_Y = read_tensors(str(Config.TRAIN_FILE_PATH))
    mse_train, rmse_train, r2_train = calculate_metrics(model, train_X, train_Y)
    
    with open(str(Config.METRICS_FILE_PATH), 'w') as metric_file:
        json.dump(dict( r_squared_train=r2_train, 
                        mse_train=mse_train,
                        rmse_train=rmse_train,
                        r_squared_test=r2_test,
                        mse_test=mse_test,
                        rmse_test=rmse_test), metric_file)

def load_model() -> object:
    """Function for loading the model.

    Returns:
        object: Returns the model for the inferecing.
    """
    checkpoint = torch.load(str(Config.MODEL_FILE_PATH))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state'])
    for parameter in model.parameters():
        parameter.requires_grad = False    
    logger.info('Model is loaded')
    return model

def create_forecast_plot(model: object) -> None:
    """Function for creating the forecast plots.

    Args:
        model (object): Model for performing inferencing.
    """
    model.eval()
    predict_X, predict_Y = read_tensors(str(Config.PREDICT_FILE_PATH))
    train_predict = model(predict_X)

    data_predict = train_predict.data.numpy()
    predict_Y_plot = predict_Y.data.numpy()
    
    sc = MinMaxScaler()    
    sc.fit_transform(data_predict)

    data_predict = sc.inverse_transform(data_predict)
    predict_Y_plot = sc.inverse_transform(predict_Y_plot)

    #plt.axvline(x=int(), c='r', linestyle='--')

    plt.plot(predict_Y_plot)
    plt.plot(data_predict)
    plt.suptitle('Time-Series Prediction')
    plt.show()
    plt.savefig(str(Config.PLOT_FILE_PATH), dpi=200)
    logger.info('Forecast plot is created.')

def main_evaluate() -> None:
    """Function for calling the other services.
    """
    model = load_model()
    create_forecast_plot(model)
    perform_save_metrics(model)



if __name__ == '__main__':    
    global logger
    logging.basicConfig(level = logging.DEBUG, filemode='a')
    file_handler = logging.FileHandler('log/evlaluate_model.log')
    logger = logging.getLogger('evlaluate_model')
    logger.addHandler(file_handler)
    main_evaluate()
