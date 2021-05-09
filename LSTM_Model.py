import torch

from torch import nn
from torch.nn.utils import rnn

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size):

        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(hidden_size, 128)
        self.bn = nn.BatchNorm1d(700)

        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)

        self.linear2 = nn.Linear(128, 2)
    
        #affine, bn, relu, do

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        dropout = self.dropout(lstm_out)
        linear1 = self.linear1(dropout)
        normd = self.bn(linear1)
        relud = self.relu(normd)
 
        dropped = self.dropout2(relud)

        pred = self.linear2(dropped)
        
        return pred