import os
import sys
from easydict import EasyDict
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))

from utils import utils
from model.basemodel import Model
from model import bert, lstm, wave2vec, multimodal, likelihood


def get_model(config: EasyDict) -> Model:
    """ Get the model according the configuration (config) """
    implemented = ['text', 'audio', 'video', 'multi']
    if config.task not in implemented:
        raise NotImplementedError(f'Expected config.task in {implemented} but found {config.task}')

    cfg_model = config.model[config.task]

    if config.task == 'text':
        model = bert.BertClassifier(hidden_size=cfg_model.hidden_size,
                                    num_classes=config.data.num_classes,
                                    pretrained_model_name=cfg_model.pretrained_model_name,
                                    last_layer=True)
    
    if config.task == 'audio':
        model = wave2vec.Wav2Vec2Classifier(pretrained_model_name=cfg_model.pretrained_model_name,
                                            last_layer=True,
                                            num_classes=config.data.num_classes,
                                            audio_length=config.data.audio_length)
    
    if config.task == 'video':
        model = lstm.LSTMClassifier(num_features=config.data.num_features,
                                    hidden_size=cfg_model.hidden_size,
                                    num_classes=config.data.num_classes,
                                    last_layer=True)
    
    if config.task == 'multi':
        basemodel = {}
        if config.load.text[0]:
            text_config = utils.load_config_from_folder(path=config.load.text[1])
            text = bert.BertClassifier(hidden_size=text_config.model.text.hidden_size,
                                       pretrained_model_name=text_config.model.text.pretrained_model_name,
                                       last_layer=True)
            utils.load_weigth(text, logging_path=config.load.text[1])
            basemodel['text'] = text
        
        if config.load.audio[0]:
            audio_config = utils.load_config_from_folder(path=config.load.audio[1])
            audio = wave2vec.Wav2Vec2Classifier(pretrained_model_name=audio_config.model.audio.pretrained_model_name,
                                                last_layer=True,
                                                audio_length=audio_config.data.audio_length)
            utils.load_weigth(audio, logging_path=config.load.audio[1])
            basemodel['audio'] = audio
        
        if config.load.video[0]:
            video_config = utils.load_config_from_folder(path=config.load.video[1])
            video = lstm.LSTMClassifier(num_features=video_config.data.num_features,
                                        hidden_size=video_config.model.video.hidden_size,
                                        last_layer=True)
            utils.load_weigth(video, logging_path=config.load.video[1])
            basemodel['video'] = video
        
        if config.model.multi.likelihood:
            model = likelihood.Likelihood(basemodel=basemodel)
        else:
            model = multimodal.MultimodalClassifier(basemodel=basemodel,
                                                    last_hidden_size=config.model.multi.hidden_size,
                                                    freeze_basemodel=config.model.multi.freeze_basemodel,
                                                    num_classes=config.data.num_classes)
    
    return model


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    from icecream import ic 
    import torch 

    config = EasyDict(yaml.safe_load(open('config/config.yaml', mode='r')))
    ic(config)

    model = get_model(config)
    ic(model)
    model.train()
    audio = torch.rand((32, 1000, 2))
    print(f'input shape: {audio.shape}')
    
    y = model.forward(x=audio)
    print(f'output shape: {y.shape}')

    # device = torch.device("cuda")
    # audio = audio.to(device)
    # model = model.to(device)
    # # model.check_device()

    # audio = audio.to(device)
    # model.forward(x=audio)