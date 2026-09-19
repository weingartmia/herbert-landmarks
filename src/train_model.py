
import torch
from torch import nn
from models.base_model import herbert
from data.load_data import test_loader,train_loader, rot_test_loader,rot_train_loader

learning_rate= 0.0001
batch_size=32
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def train_loop_distance(dataloader, model,loss_fn, optimizer):
    size = len(dataloader.dataset)
    

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(f"traning {batch}")

       
        pred,_ = model(X)
        pred= pred.squeeze(1)

        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward() # compute the negative gradient
        optimizer.step()# increase/decrease weight
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train_loop_rotation(dataloader, model,loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(f"training rotation {batch}")
       
        # Compute prediction error
        _,pred = model(X)
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
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            
            pred,_ = model(X)
            pred= pred.squeeze(1)
            test_loss += loss_fn(pred, y).item()
            mae = torch.abs(pred - y).mean()
            print(f" Prediction cm: {pred}, Real values cm : {y}, difference {torch.abs(pred-y)} ")
        
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    # correct /= size
    
    print(f"Test Error: Avg loss: {test_loss:>8f}, Avg distance loss: {mae} cm \n")

def test_loop_rotation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            print(y)
            _,pred = model(X)
            print(pred)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_model():
    loss_fn = torch.nn.MSELoss()
    loss_fn_rot= nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(herbert.parameters(), lr=learning_rate)# learning rate is new w= old w -(lr * gradient) basically how much 

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loop_distance(train_loader, herbert,loss_fn, optimizer)
        test_loop_distance(test_loader, herbert, loss_fn)

        train_loop_rotation(rot_train_loader,herbert,loss_fn_rot,optimizer)
        test_loop_rotation(rot_test_loader,herbert,loss_fn_rot)


    print("Done!")

if __name__=="__main__":
    train_model()
    torch.save(herbert.state_dict(), "results/herbert.pth")