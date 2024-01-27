import os
import sys
import pandas as pd
from typing import Tuple
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.nn.functional import one_hot
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.dataloader import get_data

torch.random.seed()


class DataGenerator(Dataset):
    def __init__(self,
                 mode: str,
                 data_path: str,
                 load: dict[str, bool],
                 sequence_size: int,
                 audio_size: int,
                 video_size: int) -> None:
        super().__init__()

        print(f'creating {mode} generator')
        self.mode = mode
        self.data_path = data_path
        self.load = load

        self.sequence_size = sequence_size
        self.audio_size = audio_size
        self.video_size = video_size

        self.df = pd.read_csv(os.path.join(data_path, f"item_{mode}.csv"))
        self.num_data = len(self.df)

        self.num_line_to_load_for_text = 8

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self) -> int:
        return self.num_data
    
    def __getitem__(self, index: int) -> Tuple[dict[str, Tensor], Tensor]:
        """
        ----
        ARGUMENTS
        index: int
            number of batch that will be load
        -----
        OUTPUT:
        data: dict[str, Tensor]
            dictionary of data witch containt keys like text, audio or/and video
            and the torch tensor for the value of the corresponding key
            find the shape below
        label: Tensor
            label of the results

        -----
        SHAPE AFTER BATCH
        text  shape:    (batch_size, sequence size)
        audio shape:    (batch_size, audio_length, 2)
        video shape:    (batch_size, num_frames, num_features, 2)
        label shape:    (batch_size, num_classes)
        """
        line = self.df.loc[index]
        label = torch.tensor(int(line['label']))
        label = one_hot(label, num_classes=2).to(torch.float32)
        data : dict[str, Tensor] = {}

        if self.load['text']:
            text = get_data.get_text(info=line, num_line_to_load=self.num_line_to_load_for_text)
            text = self.tokenizer(text)['input_ids'][:self.sequence_size]

            if len(text) < self.sequence_size:
                raise ValueError(f'text must be have more than {self.sequence_size} elements, but found only {len(text)} elements.\n'
                                 f"the file is: {line['text_filepath']} in the line (ipu)={line['ipu_id']}. Number line to load is {self.num_line_to_load_for_text}.\n"
                                 f"{self.mode = }, {index = }")
            
            text = torch.tensor(text)
            data['text'] = text
        
        if self.load['audio']:
            audio = get_data.get_audio_sf(info=line, audio_length=self.audio_size)
            data['audio'] = audio

            if audio.shape != (self.audio_size, 2):
                raise ValueError(f'audio shape expected {(self.audio_size, 2)} but found {audio.shape}.\n'
                                 f'Change the audio size or remove the item {line["item"]} in item_{self.mode}.csv')

        if self.load['video']:
            s0 = get_data.get_frame(info=line, video_size=self.video_size, speaker=0)
            s1 = get_data.get_frame(info=line, video_size=self.video_size, speaker=1)

            video = torch.stack([s0, s1], dim=len(s0.shape))
            data['video'] = video
        
        return data, label
        


def create_dataloader(mode: str, config: EasyDict) -> DataLoader:
    """ Create a dataloader 
    -----
    ARGUMENTS
    mode: str
        select mode. Can be train, val or test
    config: EasyDict    
    -----
    OUTPUTS:
    dataloader: DataLoader
    """
    if mode not in ['train', 'val', 'test']:
        raise ValueError(f"mode must be train, val or test but is '{mode}'")

    if config.task != 'multi':
        load = dict(map(lambda x: (x, config.task == x), ['text', 'audio', 'video']))
    else:
        load = dict(map(lambda x: (x[0], x[1][0]), config.load.items()))

    generator = DataGenerator(mode=mode,
                              data_path=config.data.path,
                              load=load, 
                              sequence_size=config.data.sequence_size,
                              audio_size=config.data.audio_length,
                              video_size=config.data.num_frames)
       
    dataloader = DataLoader(generator,
                            batch_size=config.learning.batch_size,
                            shuffle=True,
                            drop_last=True)
    return dataloader




if __name__ == '__main__':
    import yaml
    from icecream import ic

    stream = open('config/config.yaml', 'r')
    config = EasyDict(yaml.safe_load(stream))
    config.task = 'all'
    ic(config)

    test_dataloader = create_dataloader(mode='test', config=config)
    data, label = next(iter(test_dataloader))
    
    print('text  shape:', data['text'].shape)
    print('audio shape:', data['audio'].shape)
    print('video shape:', data['video'].shape)
    print('label shape:', label.shape)

    # for mode in ['train', 'val', 'test']:

    #     generator = DataGenerator(mode=mode,
    #                               data_path='data', 
    #                               load={'audio': True, 'text': False, 'video': False},
    #                               audio_size=2000,
    #                               sequence_size=20,
    #                               video_size=10)
        
    #     for i in range(len(generator)):
    #         generator.__getitem__(index=i)


    # generator = DataGenerator(mode='train',
    #                           data_path='data', 
    #                           load={'audio': False, 'text': False, 'video': False},
    #                           audio_size=2000,
    #                           sequence_size=20,
    #                           video_size=10)

    # sum_label = torch.zeros((2))
    # ic(sum_label.shape)
    # for i in range(len(generator)):
    #     sum_label += generator.__getitem__(index=i)[-1]
    
    # ic(sum_label / sum(sum_label))
