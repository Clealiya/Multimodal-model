import os
import sys
from typing import Any
from os.path import dirname as up

import torch
from torch import nn, Tensor
from transformers import BertModel  

sys.path.append(up(up(os.path.abspath(__file__))))

from model.basemodel import BaseModel


class BertClassifier(BaseModel):
    def __init__(self,
                 hidden_size: int=768,
                 num_classes: int=2,
                 pretrained_model_name: str='camembert-base', 
                 last_layer: bool=True,
                 freeze_bert_parameters: bool=True
                 ) -> None:
        super(BertClassifier, self).__init__(hidden_size, last_layer, num_classes)

        self.bert=BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)

        if freeze_bert_parameters:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0)
        self.fc = nn.Linear(in_features=768, out_features=hidden_size)
        self.last_linear = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()

        # print(next(self.fc.parameters()))

    def forward(self,
                x: Tensor,
                attention_mask: Any=None
                ) -> Tensor:
        """
        x shape: (B, sequence_size),                dtype: torch.int64
        output_shape: (B, C) or (B, hidden_size)    dtype: torch.float32
        """
        outputs = self.bert(x, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:,0,:]
        x = self.relu(x)
        logits = self.fc(x)

        if self.last_layer:
            x = self.dropout(x)
            logits = self.forward_last_layer(x=logits)
        
        return logits
    

if __name__ == '__main__':
    from icecream import ic
    model = BertClassifier()

    state_dict = model.get_only_learned_parameters()
    ic(state_dict)

    x = torch.randint(0, 10, (64, 20))
    print(x.shape)
    y = model(x)
    print(y.shape)

    # torch.save(state_dict, 'textmodel_weigth.pt')

    # state_dict = torch.load('textmodel_weigth.pt')
    # ic(state_dict)

    # model.load_state_dict(state_dict, strict=False)

