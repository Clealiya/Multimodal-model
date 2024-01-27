import os
import sys
import yaml
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from src.model.basemodel import Model


def forward(model: Model, data: dict[str, Tensor], task: str) -> Tensor:
    """ forward data in the model accroding the task """
    if task != 'multi':
        return model.forward(data[task])
    else:
        return model.forward(data)


def dict_to_device(data: dict[str, Tensor], device: torch.device) -> None:
    """ load all the data.values in the device """
    for key in data.keys():
        data[key] = data[key].to(device)


def get_device(device_config: str) -> torch.device:
    """ get device: cuda or cpu """
    if torch.cuda.is_available() and device_config == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def load_weigth(model: torch.nn.Module, logging_path: str) -> None:
    checkpoint_path = os.path.join(logging_path, 'checkpoint.pt')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Error: model weight was not found in {checkpoint_path}')
    problem = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print(problem)


def load_config_from_folder(path: str) -> EasyDict:
    """ load the config.yaml in the folder """
    file = os.path.join(path, 'config.yaml')
    if not os.path.exists(file):
        raise FileNotFoundError(f'config system was not found in {file}')
    
    stream = open(file, 'r')
    return EasyDict(yaml.safe_load(stream))


def is_model_likelihood(config: EasyDict) -> bool:
     """ test if the configuration is a likelihood model """
     return config.task == 'multi' and config.model.multi.likelihood
