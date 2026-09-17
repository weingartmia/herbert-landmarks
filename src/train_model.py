
import torch
from torch import nn
from models.base_model import herbert
from data.load_data import test_loader,train_loader, rot_test_loader,rot_train_loader

learning_rate= 0.0001
batch_size=32

def train_loop(dataloader, model, loss_fn, optimizer, isDis):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print("started training")

        if isDis: pred,_ = model(X).squeeze(1)
        else: _,pred= model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() # compute the negative gradient
        optimizer.step()# increase/decrease weight
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, isDis):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            
            if isDis:pred,_ = model(X).squeeze(1)
            else: _,pred= model(X)
            
            test_loss += loss_fn(pred, y).item()
            mae = torch.abs(pred - y).mean()
            print(f" Prediction cm: {pred}, Real values cm : {y}, difference {torch.abs(pred-y)} ")
            
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    # correct /= size
    
    print(f"Test Error: Avg loss: {test_loss:>8f}, Avg distance loss: {mae} cm \n")


def train_model():
    loss_fn = torch.nn.MSELoss()
    loss_fn_rot= nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(herbert.parameters(), lr=learning_rate)# learning rate is new w= old w -(lr * gradient) basically how much 

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loop(train_loader, herbert, loss_fn, optimizer, True)
        test_loop(test_loader, herbert, loss_fn,True)

        train_loop(rot_train_loader,herbert,loss_fn_rot,optimizer,False)
        test_loop(rot_test_loader,herbert,loss_fn_rot,False)


    print("Done!")

if __name__=="__main__":
    train_model()
    torch.save(herbert.state_dict(), "results/herbert.pth")