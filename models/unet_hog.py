import torch.nn as nn

##### HOG_Decoder ####
class UNet_HOG_Decoder(nn.Module):

    def __init__(self, hog_dim):
        super(UNet_HOG_Decoder,self).__init__()
        
        self.hog = nn.Sequential(
            nn.Conv2d(1024, 512,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2, ceil_mode=True),
            nn.Conv2d(512, 256,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(4096, hog_dim)
        )

        

    def forward(self, z):

        return self.hog(z)