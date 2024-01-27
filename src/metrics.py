import os
import sys
import numpy as np
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torchmetrics import Accuracy, F1Score, Precision, Recall

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))


class Metrics:
    def __init__(self, config: EasyDict) -> None:
        if config.data.num_classes != 2:
            raise NotImplementedError(f'Attention: only binary accuracy was implemented')
        
        self.metrics = {}
        metrics_name = []
        # test if metrics is in config
        if 'metrics' in config:
            metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))

        task = 'multiclass'
        average = 'macro'
        
        # test with metrics there are
        if 'acc' in metrics_name:
            self.metrics['acc'] = Accuracy(task='binary')
        
        if 'precision' in  metrics_name:
            self.metrics['precision'] = Precision(task=task,
                                                  num_classes=2,
                                                  average=average)
        
        if 'recall' in metrics_name:
            self.metrics['recall'] = Recall(task=task,
                                            num_classes=2,
                                            average=average)
        
        if 'f1' in  metrics_name or 'f1score' in metrics_name:
            self.metrics['f1'] = F1Score(task=task,
                                         num_classes=2,
                                         average=average)
    
        self.num_metrics = len(self.metrics)
        self.metrics_name = list(self.metrics.keys())
    
    def compute(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        """ compute all the metrics 
        y_pred and y_true must have shape like (B, 2)
        """
        metrics_value = []
        y_true = torch.argmax(y_true, dim=-1)
        y_pred = torch.argmax(y_pred, dim=-1)
        for metric in self.metrics.values():
            metrics_value.append(metric(y_pred, y_true).item())
        return np.array(metrics_value)

    def __str__(self) -> str:
        return f'Metrics: {self.metrics}'
    
    def to(self, device: torch.device) -> None:
        for key in self.metrics.keys():
            self.metrics[key] = self.metrics[key].to(device)
    
    def table(self, metrics_value: np.ndarray) -> str:
        if len(metrics_value) != self.num_metrics:
            ValueError(f'Expected metrics_value have size {self.num_metrics} but found {len(metrics_value)}.')
        
        table = f'   METRICS NAME{" " * 5} -> METRICS VALUE\n'
        for i in range(self.num_metrics):
            metric_name = self.metrics_name[i]
            table += f"{metric_name}{' ' * (20 - len(metric_name))} -> {metrics_value[i]:.3f}\n"
        
        return table



if __name__ == '__main__':
    import yaml
    config = EasyDict(yaml.safe_load(open('config/config.yaml', 'r')))

    B = 128
    seuil = 0.8
    y_true = torch.rand((B, 2))
    y_pred = torch.rand((B, 2))
    y_pred_0 = torch.zeros((B, 2))
    y_pred_1 = torch.zeros((B, 2))
    for i in range(B):
        y_pred_1[i, 1] = 1
    y_true[y_true <= seuil] = 0 
    y_true[y_true > seuil] = 1

    metrics = Metrics(config=config)
    print(metrics)

    print('y_pred rand:')
    metrics_value = metrics.compute(y_pred=y_pred, y_true=y_true)
    print(metrics.table(metrics_value))

    print('y_pred 0:')
    metrics_value = metrics.compute(y_pred=y_pred_0, y_true=y_true)
    print(metrics.table(metrics_value))

    print('y_pred 1:')
    metrics_value = metrics.compute(y_pred=y_pred_1, y_true=y_true)
    print(metrics.table(metrics_value))
        
