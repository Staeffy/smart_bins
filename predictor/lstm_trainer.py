import logging
import torch
from config import Config
from lstm_model import LSTM
from typing import Tuple
from torch.autograd import Variable
import numpy as np
import torch.tensor as tensor

def read_features() -> Tuple[torch.Tensor, torch.Tensor]:

    
    for item_number, tensor in enumerate(torch.load(str(Config.TRAIN_FILE_PATH))):

        if item_number == 0:
            train_X = tensor
        elif item_number == 1:
            train_Y = tensor

    loaded = torch.load(str(Config.TRAIN_FILE_PATH))

    return train_X, train_Y   



def trainer_main() -> None:

    number_of_classes = int(Config.NUMBER_OF_CLASSES)
    number_of_layers = int(Config.NUMBER_OF_LAYERS)
    input_size = int(Config.INPUT_SIZE)
    hidden_size = int(Config.HIDDEN_SIZE)
    sequence_length = int(Config.SEQ_LENGTH)

    train_X, train_Y = read_features()
    lstm = LSTM(number_of_classes, input_size, hidden_size, number_of_layers)
    logger.info(type(train_X))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=int(Config.LEARNING_RATE))

    for epoch in range(int(Config.NUMBER_EPOCHS)):
        outputs = lstm(train_X)
        optimizer.zero_grad()
        
        loss = criterion(outputs, train_Y)
        
        loss.backward()
        
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

if __name__ == '__main__':
    global logger
    logging.basicConfig(level = logging.DEBUG, filemode='a')
    file_handler = logging.FileHandler('log/create_features.log')
    logger = logging.getLogger('log_file')
    logger.addHandler(file_handler)
    trainer_main()