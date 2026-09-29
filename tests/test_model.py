import torch

import transforms.transforms as t

import cv2



def sanity_check(optimizer,model,criterion, loader):
    images, labels = next(iter(loader))
    for epoch in range(1000):

        optimizer.zero_grad()
       
        output = model(images)
      

        loss = criterion(output.squeeze(), labels)

        loss.backward()

        before = model.regressor[-1].weight.detach().clone()
        
        optimizer.step()

        after = model.regressor[-1].weight.detach()
                
        print("weight change:", (after - before).abs().mean().item())

        print("gradient:", model.regressor[-1].weight.grad.abs().mean().item())

        if epoch % 50 == 0:
            pred = output
            tol = 0.1
            acc = (pred == labels).float().mean()

            print(pred, labels)

            probability = torch.sigmoid(pred)  
           
            prediction = (probability).float().round()
            confidence =((probability >  0.5 + tol) | (probability < 0.5 -tol)).float()
            accurance = (prediction.squeeze() == labels).float().mean()



            print(epoch,"loss:", loss.item(),"accuracy:", accurance )
            print(f"prediction: {prediction} probability: {probability}")
            print(f"confidence  mean: {confidence.mean()}")

    print("WEIGHT:", model.regressor[-1].weight[0, :5])

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



def test_model_eval(model,dataset):

    X,y = dataset[3][0], dataset[3][1]
    X=t.transform_landmarks(X)
    X =X.unsqueeze(0)
    model.eval()

    with torch.no_grad():
        # X = X.to(device)
        pred = model(X)
        predicted, actual =[torch.sigmoid(pred), y]
        print(f'Predicted raw : "{predicted}", Predictes: {predicted.round()} Actual: "{actual}"')




def test_models_workflow(path,evaluate):

    video = cv2.VideoCapture(path)

    while True:
        ret, frame= video.read()
        # frame = cv2.flip(frame, 1)

        if not ret:
            break

        evaluate(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()       


