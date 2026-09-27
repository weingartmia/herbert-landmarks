import torch
from torch import nn



import sys
sys.path.append("..")
import tests.test_model as test_model
from models.base_model import norbert
from data.load_data import bin_test_loader,bin_train_loader,  subset_bin







learning_rate= 1e-4
batch_size=32
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")



def train_loop_binary(dataloader, model,loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        print(f"training {batch}")
       
        pred = model(X)

 
        print(y)
        loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        print("pred mean:", pred.mean().item())
        print("pred std :", pred.std().item())
        print("label mean:", y.float().mean().item())
        print("loss:", loss.item())
        print("pred std :", pred.std().item())
        print("label std:", y.std().item())

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




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
            tol = 0.1
            prediction = (probability).float().round()
            confidence =((probability >  0.5 + tol) | (probability < 0.5 -tol)).float()
            accurance = (prediction.squeeze() == y).float().mean()

            print(pred)
            test_loss += loss_fn(pred.squeeze(), y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            
            print("accuracy:", accurance )
            print(f"prediction: {prediction} probability: {probability}")
            print(f"confidence  mean: {confidence.mean()}")

            print(f" predicted value: {prediction}, probbability: {probability} ")
    test_loss /= num_batches
    # correct /= size
    print(f" Avg loss: {test_loss:>8f} \n")



def train_loop():


    epochs=100
    loss_binary=nn.BCEWithLogitsLoss()

    optimizer= torch.optim.Adam(norbert.parameters(), lr=learning_rate)
    # test_model.sanity_check(optimizer,norbert,loss_binary,subset_bin)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loop_binary(bin_train_loader,norbert, loss_binary,optimizer)
        test_loop_binary(bin_test_loader,norbert,loss_binary)

if __name__=="__main__":
    train_loop()
    torch.save(norbert.state_dict(), "results/norbert.pth")