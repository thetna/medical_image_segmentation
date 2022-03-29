
import torch.nn as nn
import torch.nn.functional as F



##### HOG_Decoder ####
class U2Net_HOG_Decoder(nn.Module):

    def __init__(self, hog_dim):
        
        super(U2Net_HOG_Decoder,self).__init__()

        self.hog1 = nn.Sequential(
            nn.Conv2d(64, 128,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,stride=4, ceil_mode=True),
            nn.Conv2d(128, 256,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,stride=4, ceil_mode=True),
            nn.Conv2d(256, 256,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(16384,hog_dim)
        )
        
        self.hog2 = nn.Sequential(
            nn.Conv2d(64, 256,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,stride=4, ceil_mode=True),
            nn.Conv2d(256, 256,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,stride=4, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(16384,hog_dim)
        )
        
        self.hog3 = nn.Sequential(
            nn.Conv2d(128, 256,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,stride=4, ceil_mode=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(16384, hog_dim)
        )
        
        self.hog4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4,stride=4, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(16384, hog_dim)
        )
        
        self.hog5 = nn.Sequential(
            nn.Conv2d(512, 256,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(16384,hog_dim)
        )
        
        self.hog6 = nn.Sequential(
            nn.Conv2d(512, 256,3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16384,hog_dim)
        )

        

    def forward(self, hx1d, hx2d, hx3d, hx4d, hx5d, hx6):

        return self.hog1(hx1d), self.hog2(hx2d), self.hog3(hx3d), self.hog4(hx4d), self.hog5(hx5d), self.hog6(hx6)
