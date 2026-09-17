import sys
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.io import decode_image

sys.path.append("..")
from transforms.transforms import transform_distance,transform_rotation

class DistanceDataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
      

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        distance = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
 
        return image, distance

class RotationDataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
      

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
       
        if self.transform:
            image = self.transform(image)
 
        return image, label

full_dataset= DistanceDataset(
    annotations_file='data/dataset/labels/distance.csv',
    img_dir="data/dataset/images",
    transform= transform_distance

)

rotation_dataset= RotationDataset(
    annotations_file="data/dataset/labels/rotations.csv",
    img_dir="data/dataset/images",
    transform=transform_rotation

)

train_size = int(0.7 * len(full_dataset)) # 70% of all datasets are for training
test_size = int(0.3 * len(full_dataset)) # 15% for 

rot_train_size=int(0.7* len(rotation_dataset))
rot_test_size= int(0.3 * len(rotation_dataset))

training_dataset,test_dataset=random_split(full_dataset,[train_size,test_size])

rot_training_dataset, rot_test_dataset= random_split(rotation_dataset,[rot_train_size,rot_test_size])

print(f"training datasets: {training_dataset.indices}")
print(f"test_dataset: {test_dataset.indices}")
