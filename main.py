import os
import argparse
from icecream import ic

from config.config import load_config, find_config, train_logger
from src.train import train
from src.test import test
from utils import utils

import os


def main(options: dict) -> None:
    
    config = load_config(options['config_path'])

    if options['mode'] == 'train':
        if options['task'] is not None:
            config.task = options['task']
        ic(config)
        train(config)
    
    if options['mode'] == 'test':
        
        if options['path'] is None:
            raise ValueError(f'you must specify the path of the experiment that you want to test')
        
        config = load_config(find_config(experiment_path=options['path']))
        ic(config)
        test(config, logging_path=options['path'])
    
    if options['mode'] == 'latefusion':
        config.task = 'likelihood'
        config.model.multi.likelihood = True
        logging_path = train_logger(config=config, write_train_log=False)
        config.task = 'multi'
        ic(logging_path)
        test(config=config,
             logging_path=logging_path)

    
    if options['mode'] == 'baseline':
        config = load_config(options['config_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', '-m', default=None, type=str,
                        choices=['train', 'test', 'latefusion'],
                        help="choose a mode between 'train', 'test'")
    parser.add_argument('--config_path', '-c', default=os.path.join('config', 'config.yaml'),
                        type=str, help="path to config (for training)")
    parser.add_argument('--path', '-p', type=str,
                        help="experiment path (for test, prediction or generate)")
    parser.add_argument('--task', '-t', type=str, default=None,
                        choices=['text', 'video', 'audio', 'multi'],
                        help="task for model (will overwrite the config) for trainning")

    args = parser.parse_args()
    options = vars(args)

    main(options)