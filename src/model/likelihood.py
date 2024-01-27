import os
import sys
from os.path import dirname as up

import torch    

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from model.basemodel import Model, BaseModel


class Likelihood(Model):
    def __init__(self,
                 basemodel: dict[str, BaseModel],
                 ) -> None:
        super().__init__()

        self.keys = list(basemodel.keys())

        if not all(element in ['text', 'video', 'audio'] for element in self.keys):
            raise ValueError(f"all keys of basemodel must be text, audio or video. But the keys are {list(basemodel.keys())}")

        self.basemodel = basemodel
        self.n = len(self.basemodel)
        print(f'Multimodal model which take {self.keys}')

        for model in self.basemodel.values():
            model.put_last_layer(last_layer=True)
            for param in model.parameters():
                param.requires_grad = False


    def forward(self,
                data: dict[str, torch.Tensor]
                ) -> torch.Tensor:
        """ compute the mean of the baseline output

        input       shape                          dtype
        text    (B, sequence_size)              torch.int64
        audio   (B, audio_length)               torch.float32
        frames  (B, num_frames, num_features)   torch.float32

        ouput       shape                          dtype
        logits  (B, num_classes)                torch.float32
        """
        x = 0
        for key in self.keys:
            x += self.basemodel[key].forward(data[key])
        
        return x / self.n
    
    def to(self, device: torch.device):
        super().to(device)
        for model in self.basemodel.values():
            model = model.to(device)
        return self
        


