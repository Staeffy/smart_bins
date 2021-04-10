import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from config import Config

class LSTM(nn.Module):

    def __init__(self, number_of_classes, input_size, hidden_size, number_of_layers) -> None:
        
        super(LSTM, self).__init__()
        
        self.number_of_classes = number_of_classes
        self.number_of_layers = number_of_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = int(Config.SEQ_LENGTH)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=number_of_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, number_of_classes)

    def forward(self, x):

        h_0 = Variable(torch.zeros(
            self.number_of_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.number_of_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))        
        h_out = h_out.view(-1, self.hidden_size)        
        out = self.fc(h_out)

        return out
    