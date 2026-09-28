from models.base_model import Herbert,Norbert
import transforms.transforms as t


import torch
from torch import nn

import cv2


import sys 

sys.path.append("..")
import tests.test_model as test_model
from data.distance_datasets import bin_test_dataset
from utils.show_coordinates import show_absolute_coordinates



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

herbert = Herbert().to(device)
norbert =Norbert().to(device)
herbert.load_state_dict(torch.load("results/herbert.pth", weights_only=True))
norbert.load_state_dict(torch.load("results/norbert.pth", weights_only=True))


def is_hand(x):

    norbert.eval()
    # frame= t.transform_landmarks(frame)
    # x=frame.unsqueeze(0)

    with torch.no_grad():
        x= x.to(device)
        pred= norbert(x)
        prediction = torch.sigmoid(pred)
        print(f'Predicted raw : "{prediction}", Predictes: {prediction.round()}')

    if prediction.round()==1:
        return True
    return False


    
def get_landmarks(x):
    herbert.eval()
  
    with torch.no_grad():
        x= x.to(device)
        pred= herbert(x)
        print(f"landmarks: {pred}")
        

        return pred


def evaluate(frame):
    img = cv2.imread(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img,(224,224))
    


    frame= t.transform_landmarks(img)
    x=frame.unsqueeze(0)
    
    if is_hand(x): 
        print("hand detected")
        cor =get_landmarks(x)

        show_absolute_coordinates(cor.squeeze().tolist(),img)
        cv2.imshow("img",img)


if __name__ =="__main__":

    evaluate("data/dataset/images/landmarks/000001.jpg")
    # evaluate(bin_test_dataset[5][0])
    # test_model.test_model_eval(norbert,bin_test_dataset)

    cv2.waitKey(0)
    cv2.destroyAllWindows()