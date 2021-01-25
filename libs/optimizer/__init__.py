import math
from typing import Any, Tuple

import torch.nn as nn
import torch.optim as optim

__all__ = ["get_optimizer"]


def get_optimizer(model: nn.Module, max_epoch: int) -> Tuple[Any, Any]:
    optimizer = optim.SGD([
        {'params': model.feature_conv.parameters(), 'lr': 1e-3},
        {'params': model.feature_res_1.parameters(), 'lr': 1e-3},
        {'params': model.feature_res_2.parameters(), 'lr': 1e-3},
        {'params': model.feature_dilated_res_1.parameters(), 'lr': 1e-3},
        {'params': model.feature_dilated_res_2.parameters(), 'lr': 1e-3},
        {'params': model.pyramid_pooling.parameters(), 'lr': 1e-3},
        {'params': model.decode_feature.parameters(), 'lr': 1e-2},
        {'params': model.aux.parameters(), 'lr': 1e-2},
    ], momentum=0.9, weight_decay=0.0001)
    
    lr_lambda = lambda_epoch(max_epoch)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler

class lambda_epoch():
    def __init__(self, max_epoch: int = 30) -> None:
        self.max_epoch = max_epoch

    def __call__(self, epoch):
        return math.pow((1-epoch/self.max_epoch), 0.9)
