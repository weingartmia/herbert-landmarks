from models.base_model import NeuralNetwork
import transforms.transforms as t

import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

herbert = NeuralNetwork().to(device)
herbert.load_state_dict(torch.load("results/herbert.pth", weights_only=True))


def evaluate(frame):
    herbert.eval()
    frame= t.transform_distance(frame)
    x=frame.unsqueese(0)
    with torch.no_grad():
        x= x.to(device)
        pred= herbert(x).squeese(1)
        print(pred.item())
        return pred.item()
