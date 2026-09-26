import torch


def sanity_check(optimizer,model,criterion, loader):
    images, labels = next(iter(loader))
    for epoch in range(1000):

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            pred = output.argmax(dim=1)
            acc = (pred == labels).float().mean()

            print(epoch,"loss:", loss.item(),"accuracy:", acc.item() )

def sanity_test_landmarks(optimizer,model,criterion, loader):

    images, labels = next(iter(loader))
    for epoch in range(100):

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)


        before = model.regressor[-1].weight.detach().clone()

        loss.backward()

      
        optimizer.step()

        after = model.regressor[-1].weight.detach()
        
        print("weight change:", (after - before).abs().mean().item())

        print("gradient:", model.regressor[-1].weight.grad.abs().mean().item())
     
        if epoch % 50 == 0:
           
            tol = 0.1  # 3% of image size
            print("pred:")
            print(output)

            print("labels:")
            print(labels)
           
            dist = torch.linalg.norm(output - labels, dim=2)
            print(f" difference: {dist}")
            acc = (dist < tol).float().mean()
            

            print(epoch,"loss:", loss.item(),"accuracy:", acc.item() )
    print("WEIGHT:", model.regressor[-1].weight[0, :5])

    with torch.no_grad():
        print(model(images))

    with torch.no_grad():
        print(model(images))

def replace_test_landmarks(model,images):
    with torch.no_grad():
        a = model(images[0:1])
        b = model(images[1:2])

        print(a)
        print(b)