import os
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class Herbert(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            block(3, 32),     # 224 -> 112
            block(32, 64),    # 112 -> 56
            block(64, 128),   # 56  -> 28
            block(128, 128),  # 28  -> 14
            block(128, 128),  # 14  -> 7
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),                      # 128*7*7 = 6272
            nn.Dropout(0.3),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 8),
        )
  
    def forward(self, x):
        x = self.conv(x)
        x= self.regressor(x)

        # rotation = self.rotation_head(x)
        
        return x.view(-1,4,2)

class Norbert(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )


        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )




    def forward(self, x):
        x = self.conv(x)
        x= self.regressor(x)
   
        return x




# class Albert(nn.Module):



herbert = Herbert().to(device)

print(herbert)

norbert =Norbert().to(device)

print(norbert)