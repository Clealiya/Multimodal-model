import os
import sys
from os.path import dirname as up

import torch
from torch import nn, Tensor

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from model.basemodel import BaseModel


class LSTMClassifier(BaseModel):
    def __init__(self, 
                 num_features: int,
                 hidden_size: int,
                 num_classes: int=2,
                 last_layer: bool=True) -> None:
        """ model for learn the from video"""
        super(LSTMClassifier, self).__init__(hidden_size * 2, last_layer, num_classes)
        self.lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        """
        input shape:  (batch_size, num_frames, num_features, 2)     dtype: torch.float32
        output_shape: (B, C) or (B, hidden_size)        dtype: torch.float32
        """
        x0, x1 = x[..., 0], x[..., 1]
        output0 = self.lstm(x0)[0][:, -1, :]
        output1 = self.lstm(x1)[0][:, -1, :]

        x = torch.cat([output0, output1], dim=1)

        if self.last_layer:
            x = self.forward_last_layer(x=x)

        return x


if __name__ == '__main__':
    model = LSTMClassifier(num_features=709,
                           hidden_size=100,
                           num_classes=2,
                           last_layer=True)
    
    print('learning parameters:', model.get_number_parameters())
    x = torch.rand((64, 10, 709, 2))
    print("shape entrée:", x.shape)
    y = model.forward(x)
    print("shape sortie", y.shape)