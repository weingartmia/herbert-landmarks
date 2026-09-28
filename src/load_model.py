from models.base_model import Herbert,Norbert
import transforms.transforms as t

import torch
from torch import nn


import sys 

sys.path.append("..")
import tests.test_model as test_model
from data.load_data import bin_test_loader,bin_train_loader,  subset_bin



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

herbert = Herbert.to(device)
norbert =Norbert.to(device)
herbert.load_state_dict(torch.load("results/herbert.pth", weights_only=True))
norbert.load_state_dict(torch.load("results/norbert.pth", weights_only=True))


def evaluate(frame):

    norbert.eval()
    frame= t.transform_landmarks(frame)
    x=frame.unsqueeze(0)

    with torch.no_grad():
        x= x.to(device)
        pred= norbert(x)
        print(pred.item())
        return pred.item()
    
    print(out.item())

    if out.item()==1:
        return true
    return False

    
def get_landmarks(frame):
    x=frame
    with torch.no_grad():
        x= x.to(device)
        pred= herbert(x).squzee(1)
        print(pred.item())
        return pred.item()


if __name__ =="__main__":


    test_model.test_model_eval(norbert,bin_test_loader)
