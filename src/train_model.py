
import torch
from torch import nn
from models.base_model import herbert
from data.load_data import test_loader,train_loader,  subset


import sys
sys.path.append("..")

import tests.test_model as test_model
import utils.write_csv as write
import utils.loop_dataloader as loop

learning_rate= 1e-4
batch_size=32
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")

def train_loop_distance(dataloader, model,loss_fn, optimizer):
    size = len(dataloader.dataset)
    

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(f"traning {batch}")
        X, y = X.to(device), y.to(device)
       
        pred = model(X)

        # pred= pred.squeeze(1)
        # y= y/8.0
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward() # compute the negative gradient
      
        optimizer.step()# increase/decrease weight
        optimizer.zero_grad()

        print("pred mean:", pred.mean().item())
        print("pred std :", pred.std().item())
        print("label mean:", y.float().mean().item())
        print("loss:", loss.item())
        print("pred std :", pred.std().item())
        print("label std:", y.std().item())

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop_distance(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    n_total, correct_total,test_loss = 0, 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
      
            pred = model(X)

            test_loss += loss_fn(pred, y)

            threshold=0.1

            difference= torch.linalg.norm(pred-y, dim=2)

            correct = (difference< threshold).type(torch.float).float().mean()
            correct_total +=correct.sum().item()
            n_total += correct.numel()

            

            for i in range(4):
                acc = (difference[:, i] < threshold).float().mean()
                if (acc.item() < 0.7): write.write_values(f"landmark {i}: {acc.item()*100:.1f}%")
                print(f"landmark {i}: {acc.item()*100:.1f}%")

            print(f" predicted: {pred}")
            print(f" real: {y}")
            print(f"difference mean: {difference.mean().item()}")
            print(f"difference in pixels  {(difference.mean() * 224).item()}")


          
    test_loss /= num_batches
   
    print(f"Test Error: \n Accuracy: {(100*correct_total/n_total):>0.1f}%, Avg loss: {test_loss:>8f} \n")

            


def train_model():

    loss_smooth = nn.SmoothL1Loss(beta=0.02)

   
    optimizer= torch.optim.Adam(herbert.parameters(), lr=learning_rate)
    epochs = 100
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loop_distance(train_loader, herbert,loss_smooth, optimizer)
        test_loop_distance(train_loader, herbert, loss_smooth)


    print("Done!")



if __name__=="__main__":
    print(train_loader)
    loop.test_labels(train_loader)
    # train_model()
    # torch.save(herbert.state_dict(), "results/herbert.pth")