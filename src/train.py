import os
import sys
import time
import numpy as np
from tqdm import tqdm
from icecream import ic
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from src.metrics import Metrics
from src.model.get_model import get_model
from src.dataloader.dataloader import create_dataloader
from utils.plot_learning_curves import save_learning_curves
from utils import utils
from config.config import train_logger, train_step_logger


# LABEL_DISTRIBUTION = [0.8051, 0.1949]


def train(config: EasyDict) -> None:

    # Use gpu or cpu
    device = utils.get_device(device_config=config.learning.device)
    ic(device)

    # Get data
    train_generator = create_dataloader(config=config, mode='train')
    val_generator = create_dataloader(config=config, mode='val')
    n_train, n_val = len(train_generator), len(val_generator)
    ic(n_train, n_val)

    # Get model
    model = get_model(config)
    model = model.to(device)
    ic(model)
    ic(model.get_number_parameters())
    
    # Loss
    weight = torch.tensor([1, 3.9], device=device)
    ic(weight)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weight)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=config.learning.milesstone, gamma=config.learning.gamma)

    # Get Metrics
    metrics = Metrics(config=config)
    metrics.to(device)

    save_experiment = config.save_experiment
    ic(save_experiment)
    if save_experiment:
        logging_path = train_logger(config)
        best_val_loss = 10e6


    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        ic(epoch)
        train_loss = 0
        train_range = tqdm(train_generator)
        train_metrics = np.zeros(metrics.num_metrics)

        # Training
        model.train()
        for i, (data, y_true) in enumerate(train_range):

            utils.dict_to_device(data, device)
            y_true = y_true.to(device)
            y_pred = utils.forward(model=model, data=data, task=config.task)
                
            loss = criterion(y_pred, y_true)

            train_loss += loss.item()
            train_metrics += metrics.compute(y_pred=y_pred, y_true=y_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            current_loss = train_loss / (i + 1)
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            train_range.refresh()

        train_loss = train_loss / n_train
        train_metrics = train_metrics / n_train

        print('TRAIN:')
        print(metrics.table(train_metrics))

        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_range = tqdm(val_generator)
        val_metrics = np.zeros(metrics.num_metrics)
        
        model.eval()
        with torch.no_grad():
            
            for i, (data, y_true) in enumerate(val_range):
                
                utils.dict_to_device(data, device)
                y_true = y_true.to(device)
                y_pred = utils.forward(model=model, data=data, task=config.task)
                    
                loss = criterion(y_pred, y_true)                
                val_loss += loss.item()

                val_metrics += metrics.compute(y_pred=y_pred, y_true=y_true)

                current_loss = val_loss / (i + 1)
                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.4f}")
                val_range.refresh()
        
        scheduler.step()

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        val_loss = val_loss / n_val
        val_metrics = val_metrics / n_val

        print('VAL:')
        print(metrics.table(val_metrics))
        
        if save_experiment:
            train_step_logger(path=logging_path, 
                              epoch=epoch, 
                              train_loss=train_loss, 
                              val_loss=val_loss,
                              train_metrics=train_metrics,
                              val_metrics=val_metrics)
            
            if val_loss < best_val_loss:
                print('save model weights')
                torch.save(model.get_only_learned_parameters(),
                           os.path.join(logging_path, 'checkpoint.pt'))
                best_val_loss = val_loss

            print(f'{best_val_loss = }')

    stop_time = time.time()
    print(f"training time: {stop_time - start_time}secondes for {config.learning.epochs} epochs")

    if save_experiment and config.learning.save_learning_curves:
        save_learning_curves(logging_path)





if __name__ == '__main__':
    import yaml
    stream = open(file=os.path.join('config', 'config.yaml'), mode='r')
    config = EasyDict(yaml.safe_load(stream))

    ic(config)
    train(config=config)