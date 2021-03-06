"""Module for performing the training of LSTM model."""
import logging
import torch
from config import Config
from lstm_model import LSTM
from typing import Tuple
from torch.autograd import Variable
import numpy as np
import torch.tensor as tensor

def read_features() -> Tuple[torch.Tensor, torch.Tensor]:
    """Function for reading the torch tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Return tensors from training set.
    """

    loaded = torch.load(str(Config.TRAIN_FILE_PATH))
    train_X = loaded['train_X']
    train_Y = loaded['train_Y']

    return train_X, train_Y


def trainer_main() -> None:
    """Trainer main function.
    """

    number_of_classes = int(Config.NUMBER_OF_CLASSES)
    number_of_layers = int(Config.NUMBER_OF_LAYERS)
    input_size = int(Config.INPUT_SIZE)
    hidden_size = int(Config.HIDDEN_SIZE)

    train_X, train_Y = read_features()
    lstm = LSTM(number_of_classes, input_size, hidden_size, number_of_layers)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=float(Config.LEARNING_RATE))

    for epoch in range(int(Config.NUMBER_EPOCHS)):
        outputs = lstm(train_X)
        optimizer.zero_grad()
        
        loss = criterion(outputs, train_Y)
        loss.backward()
       
        optimizer.step()
        if epoch % 100 == 0:
            logger.info("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    
    torch.save(dict(model=lstm,model_state=lstm.state_dict()), 
          str(Config.MODEL_FILE_PATH))

    #torch.save(lstm, Config.MODEL_FILE_PATH)

if __name__ == '__main__':
    global logger
    logging.basicConfig(level = logging.DEBUG, filemode='a')
    file_handler = logging.FileHandler('log/lstm_trainer.log')
    logger = logging.getLogger('lstm_log_file')
    logger.addHandler(file_handler)
    trainer_main()