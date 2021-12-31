# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import numpy as np

class N(nn.Module):
    def __init__(self):
        super(N, self).__init__()
        self.fc1 = nn.Linear(100, 10)
        self.fc2 = nn.Linear(10, 1)


    def forward(self, x):
        return x


n = N()
k = np.array([1, -1])
mask = [1, 1]
m = np.where(np.abs(k) > 0.5, mask, np.zeros_like(k))
print(m)
