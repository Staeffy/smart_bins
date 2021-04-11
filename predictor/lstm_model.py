"""Set up module for LSTM forecast."""
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from config import Config

class LSTM(nn.Module):
    """Class for automated forecast."""

    def __init__(   self,
                    number_of_classes: int,
                    input_size: int,
                    hidden_size: int,
                    number_of_layers: int) -> None:
        """Initlializing method."""
        
        super(LSTM, self).__init__()
        
        self.number_of_classes = number_of_classes
        self.number_of_layers = number_of_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = int(Config.SEQ_LENGTH)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=number_of_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, number_of_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Inferencing from model.

        Args:
            x (torch.tensor): Batches or samples for
            training and inference.

        Returns:
            torch.tensor: Predictions.
        """


        h_0 = Variable(torch.zeros(
            self.number_of_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.number_of_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))        
        h_out = h_out.view(-1, self.hidden_size)        
        out = self.fc(h_out)

        return out
    