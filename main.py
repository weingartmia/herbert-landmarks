
from pathlib import Path

import torch
from torch import nn
import cv2
import sys 

sys.path.append("..")

from src.models.base_model import Herbert,Norbert
import src.transforms.transforms as t

import tests.test_model as test_model
from src.data.distance_datasets import bin_test_dataset
import src.utils.target_coordinates as coor



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

herbert = Herbert().to(device)
norbert =Norbert().to(device)

herbert.load_state_dict(torch.load("src/results/herbert.pth", weights_only=True))
norbert.load_state_dict(torch.load("src/results/norbert.pth", weights_only=True))

norbert.eval()
herbert.eval()


def is_hand(x):

   
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
   
    with torch.no_grad():
        x= x.to(device)
        pred= herbert(x)
        print(f"landmarks: {pred}")
        

        return pred


def evaluate(frame):


    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img,(224,224))
    
    frame= t.transform_landmarks(img)
    x=frame.unsqueeze(0)
    
    if is_hand(x): 
        print("hand detected")
        cor =get_landmarks(x)

        (dis,th,rin)=coor.get_absolute_distance(cor.squeeze().numpy())
        (rot,center)  = coor.get_rotation(cor.squeeze().numpy())

        coor.show_absolute_coordinates(cor.squeeze().tolist(),center,th,rin,img)
        

        cv2.imshow("img",img)


if __name__ =="__main__":


    
    path = Path("tests/test_assets/absolute.mp4")
    print(path.exists())
    test_model.test_models_workflow(path, evaluate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

