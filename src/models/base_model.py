import os
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class Herbert(nn.Module):

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
            nn.Linear(128, 8)
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        # self.regressor = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256 * 14 * 14, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 6)
        # )

        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128 * 28 * 28, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )

        # self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(224*224, 512),#first param is number of input and second number of output
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)# last output is the distance
        # )

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
            nn.Linear(128, 8)
        )




    def forward(self, x):
        x = self.conv(x)
        x= self.regressor(x)
   
        return x.view(-1,4,2)




# class Albert(nn.Module):



herbert = Herbert().to(device)

print(herbert)

norbert =Norbert().to(device)