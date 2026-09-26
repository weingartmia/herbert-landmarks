
import torch
from torch import nn
from models.base_model import herbert
from data.load_data import test_loader,train_loader,  subset
from data.distance_datasets import distance_map

import sys
sys.path.append("..")

import tests.test_model as test_model

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
        print(model.conv[0].weight.grad.abs().mean())
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

def train_loop_binary(dataloader, model,loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(f"training rotation {batch}")
       
        # Compute prediction error
        pred = model(X)

 
        print(y)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop_distance(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
      
            pred = model(X)

            test_loss += loss_fn(pred, y)

            threshold=0.1

            difference= torch.linalg.norm(pred-y, dim=2)

            correct += (difference< threshold).type(torch.float).float().mean()

            print(f" predicted: {pred}")
            print(f" real: {y}")
            print(torch.unique(y))


          
    test_loss /= num_batches
    correct /= size * y.shape[1]
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

            
        
    
    test_loss /= num_batches


def test_loop_binary(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            print(y)
            pred = model(X)

            probability = torch.sigmoid(pred)  
            prediction = (probability > 0.5).float()

            print(pred)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            print(f" predicted value: {prediction}, probbability: {probability} ")
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_model():
    loss_fn = torch.nn.MSELoss()
    loss_fn_rot= nn.CrossEntropyLoss()


    loss_binary=nn.BCEWithLogitsLoss()

    loss_smooth = nn.SmoothL1Loss()

    optimizer = torch.optim.SGD(herbert.parameters(), lr=learning_rate)# learning rate is new w= old w -(lr * gradient) basically how much 
    optimizer_distance= torch.optim.Adam(herbert.parameters(), lr=learning_rate)
    epochs = 100

    # test_model.sanity_test_landmarks(optimizer_distance,herbert,loss_smooth,subset)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loop_distance(train_loader, herbert,loss_smooth, optimizer_distance)
        test_loop_distance(train_loader, herbert, loss_smooth)


    print("Done!")



if __name__=="__main__":
    train_model()
    torch.save(herbert.state_dict(), "results/herbert.pth")