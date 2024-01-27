import pandas as pd
import soundfile as sf
from typing import List

import torch


def get_text(info: pd.DataFrame, num_line_to_load: int=5) -> List[str]:
    """ 
    Get the <num_line_to_load> from info['text_filepath'] and load
    return the text in list[str] where each element is a word.
    """
    filepath = info['text_filepath']
    ipu = info['ipu_id']

    df = pd.read_csv(filepath,
                     skiprows=range(1, ipu - num_line_to_load + 2),
                     nrows=num_line_to_load)
    
    text = df['text'].str.cat(sep=' ')
    return text


def get_frame(info: pd.DataFrame,
              video_size: int,
              speaker: int,
              useless_info_number: int=5
              ) -> torch.Tensor:
    """
    get the last <video_size> frame
    output shape: (<video_size>, 709)
    """
    filepath = info[f'frame_path_{speaker}']
    frame = info[f'frame_index_{speaker}']
    df = pd.read_csv(filepath,
                     skiprows=range(1, frame - video_size + 1),
                     nrows=video_size)

    colonnes_a_inclure = df.columns[useless_info_number:]
    frames = df[colonnes_a_inclure].astype('float32').to_numpy()
    frames = torch.tensor(frames)

    return frames


def get_audio_sf(info: pd.DataFrame,
                 audio_length: int
                 ) -> torch.Tensor:
    """ audio length in ms """
    end_time = int(info['stoptime'] * 1000)
    audio, _ = sf.read(file=info['audio_filepath'],
                       start=end_time - audio_length,
                       stop=end_time)
    audio = torch.tensor(audio).to(torch.float32)
    return audio


if __name__ == '__main__':
    import os
    from icecream import ic

    df = pd.read_csv(os.path.join('data', 'item_val.csv'))

    index = 31
    pd_info = df.loc[index]
    ic(pd_info)

    text = get_text(info=pd_info, num_line_to_load=5)
    ic(text)
    ic(len(text.split(' ')))