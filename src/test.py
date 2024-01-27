import os
import sys
import numpy as np
from tqdm import tqdm
from icecream import ic
from easydict import EasyDict
from os.path import dirname as up

import torch

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from src.metrics import Metrics
from src.model.get_model import get_model
from src.dataloader.dataloader import create_dataloader
from utils import utils
from config.config import test_logger


def test(config: EasyDict, logging_path: str) -> None:

    # Use gpu or cpu
    device = utils.get_device(device_config=config.learning.device)
    ic(device)

    # Get data
    test_generator = create_dataloader(config=config, mode='test')
    n_test = len(test_generator)
    ic(n_test)

    # Get model
    model = get_model(config)
    if not utils.is_model_likelihood(config):
        utils.load_weigth(model, logging_path)
    model = model.to(device)
    ic(model)
    ic(model.get_number_parameters())
    
    # Loss
    weight = torch.tensor([1, 3.9], device=device)
    ic(weight)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weight)

    # Get Metrics
    metrics = Metrics(config=config)
    metrics.to(device)

    ###############################################################
    # Start Testing                                               #
    ###############################################################

    test_range = tqdm(test_generator)
    test_metrics = np.zeros(metrics.num_metrics + 1)

    model.eval()
    with torch.no_grad():
        for i, (data, y_true) in enumerate(test_range):

            utils.dict_to_device(data, device)
            y_true = y_true.to(device)
            y_pred = utils.forward(model=model, data=data, task=config.task)
                
            loss = criterion(y_pred, y_true)

            test_metrics[0] += loss.item()
            test_metrics[1:] += metrics.compute(y_pred=y_pred, y_true=y_true)

            current_loss = test_metrics[0] / (i + 1)
            test_range.set_description(f"TEST: loss: {current_loss:.4f}")
            test_range.refresh()
        

    ###################################################################
    # Save Scores in logs                                             #
    ###################################################################
    test_metrics = test_metrics / n_test

    print(metrics.table(test_metrics[1:]))
    
    test_logger(path=logging_path,
                metrics=[config.learning.loss] + metrics.metrics_name,
                values=test_metrics)