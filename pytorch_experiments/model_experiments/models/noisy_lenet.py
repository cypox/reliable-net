import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pytorch_experiments.layers import WeightBitflipByCountLayer


class NoisyLeNet(nn.Module):
    def __init__(self, probability):
        super(NoisyLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        self.probability = probability

        self.inject = None

        #mu = probability
        #std = 0.05
        #probability = np.random.normal(mu, std)
        

    def forward(self, x):
        count = 1 if (np.random.uniform(0, 1) < self.probability) else 0
        self.inject = WeightBitflipByCountLayer(count)
        self.layer = np.random.randint(0, 11)
        
        out = self.conv1(x)
        out = self.inject(out) if self.layer == 0 else out
        out = F.relu(out)
        out = self.inject(out) if self.layer == 1 else out
        out = F.max_pool2d(out, 2)
        out = self.inject(out) if self.layer == 2 else out
        out = self.conv2(out)
        out = self.inject(out) if self.layer == 3 else out
        out = F.relu(out)
        out = self.inject(out) if self.layer == 4 else out
        out = F.max_pool2d(out, 2)
        out = self.inject(out) if self.layer == 5 else out
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.inject(out) if self.layer == 6 else out
        out = F.relu(out)
        out = self.inject(out) if self.layer == 7 else out
        out = self.fc2(out)
        out = self.inject(out) if self.layer == 8 else out
        out = F.relu(out)
        out = self.inject(out) if self.layer == 9 else out
        out = self.fc3(out)
        out = self.inject(out) if self.layer == 10 else out
        return out
