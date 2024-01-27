import os
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


def print_loss_and_metrics(train_loss: float,
                           val_loss: float,
                           metrics_name: List[str],
                           train_metrics: List[float],
                           val_metrics: List[float]) -> None:
    """ print loss and metrics for train and validation """
    print(f"{train_loss = }")
    print(f"{val_loss = }")
    for i in range(len(metrics_name)):
        print(f"{metrics_name[i]} -> train: {train_metrics[i]:.3f}   val:{val_metrics[i]:.3f}")


def save_learning_curves(path: str) -> None:
    result, names = get_result(path)

    epochs = result[:, 0]
    for i in range(1, len(names), 2):
        train_metrics = result[:, i]
        val_metrics = result[:, i + 1]
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title(names[i])
        plt.xlabel('epoch')
        plt.ylabel(names[i])
        plt.legend(names[i:])
        plt.grid()
        plt.savefig(os.path.join(path, names[i] + '.png'))
        plt.close()


def get_result(path: str) -> Tuple[List[float], List[str]]:
    with open(os.path.join(path, 'train_log.csv'), 'r') as f:
        names = f.readline()[:-1].split(',')
        result = []
        for line in f:
            result.append(line[:-1].split(','))

        result = np.array(result, dtype=float)
    f.close()
    return result, names


if __name__ == '__main__':
    logs_path = 'logs'
    # experiments_path = list(map(lambda folder: os.path.join(logs_path, folder), os.listdir(logs_path)))
    # print(experiments_path)
    # for experiment_path in experiments_path:
    #     if 'likelihood' not in experiment_path:
    #         save_learning_curves(path=experiment_path)
    
    save_learning_curves(path=os.path.join(logs_path, 'audio_1'))
