import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(512, 256,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32,3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048,1680)
        )
        
    def forward(self, x):
        return self.net(x)