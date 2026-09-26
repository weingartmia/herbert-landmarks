import sys
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.io import decode_image

import ast

sys.path.append("..")
from transforms.transforms import transform_landmarks

distance_map={
        "0.0":0,
        "0.5":1,
        "1.0":2,
        "8.0":3,
        "None":4

        }

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
        data = ast.literal_eval(label)
        
        coordinates = torch.tensor(data, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
 
        return image, coordinates

class RotationDataset(Dataset):

    rotation_map={
        "-30deg":0,
        "-15deg":1,
        "0deg":2,
        "15deg":3,
        "30deg":4

    }

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

        rotation= self.rotation_map[label]
        rotation= torch.tensor(rotation, dtype=torch.long)
       
        if self.transform:
            image = self.transform(image)
 
        return image, rotation

full_dataset= DistanceDataset(
    annotations_file='data/dataset/labels/distance.csv',
    img_dir="data/dataset/images",
    transform= transform_landmarks

)

# rotation_dataset= RotationDataset(
#     annotations_file="data/dataset/labels/rotations.csv",
#     img_dir="data/dataset/images",
#     transform=transform_rotation

# )

train_size = int(0.7 * len(full_dataset)) # 70% of all datasets are for training
test_size = len(full_dataset)- train_size# 15% for 

# rot_train_size=int(0.7* len(rotation_dataset))
# rot_test_size=  len(rotation_dataset) - rot_train_size


training_dataset,test_dataset=random_split(full_dataset,[train_size,test_size])
subset = torch.utils.data.Subset(training_dataset,range(32))
# rot_training_dataset, rot_test_dataset= random_split(rotation_dataset,[rot_train_size,rot_test_size])

print(f"training datasets: {training_dataset.indices}")
print(f"test_dataset: {test_dataset.indices}")
