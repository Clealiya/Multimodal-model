import os
import sys
from os.path import dirname as up

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from model.basemodel import BaseModel


class Wav2Vec2Classifier(BaseModel):
    def __init__(self, 
                 pretrained_model_name: str='facebook/wav2vec2-large-960h',
                 last_layer: bool=True,
                 num_classes: bool=2,
                 audio_length: int=1000
                 ) -> None:
        hidden_size = 256
        super(Wav2Vec2Classifier, self).__init__(hidden_size, last_layer, num_classes)
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        audio_length_2_hidden_size = {1000: 4096,
                                      2000: 12288,
                                      3000: 18432}
        
        if audio_length not in audio_length_2_hidden_size.keys():
            raise ValueError(f'audio_length must be in {audio_length_2_hidden_size.keys()} but found {audio_length}.')

        self.fc = torch.nn.Linear(in_features=audio_length_2_hidden_size[audio_length],
                                  out_features=hidden_size)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        pass the 2 audio into the pre-train model and concatenate the 2 outputs

        audio shape:  (B, audio_length, 2)         dtype: torch.float32
        output shape: (B, 4096) or (B, C)    dtype: torch.float32
        """
        x0, x1 = x[..., 0], x[..., 1]

        x0 = self.processor(x0, return_tensors="pt", padding="longest", sampling_rate=16000).input_values.to(self.model.device)
        x1 = self.processor(x1, return_tensors="pt", padding="longest", sampling_rate=16000).input_values.to(self.model.device)

        x0 = torch.squeeze(x0)
        x1 = torch.squeeze(x1)
        
        x0 = self.model(x0)[0]
        x0 = x0.view(x.shape[0], -1)

        x1 = self.model(x1)[0]
        x1 = x1.view(x.shape[0], -1)

        x = torch.cat([x0, x1], dim=1)

        x = self.fc(self.relu(x))

        if self.last_layer:
            x = self.relu(self.dropout(x))
            x = self.forward_last_layer(x=x)

        return x
    
    def train(self) -> None:
        """ put dropout (but not in wave2vect module) """
        if self.last_layer:
            self.dropout = self.dropout.train()
    
    def eval(self) -> None:
        if self.last_layer:
            self.dropout = self.dropout.eval()



if __name__ == '__main__':
    audio_length = 1000
    audio = torch.rand((16, audio_length, 2), dtype=torch.float32)
    print(f'input shape: {audio.shape}')
    
    model = Wav2Vec2Classifier(last_layer=True, num_classes=2, audio_length=audio_length)
    model.train()
    y = model.forward(x=audio)
    print(f'output shape: {y.shape}')

    # device = torch.device("cuda")
    # audio = audio.to(device)
    # model = model.to(device)
    # # model.check_device()

    # audio = audio.to(device)
    # model.forward(x=audio)
