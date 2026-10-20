import sys
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.io import decode_image

import ast,re

sys.path.append("..")
from transforms.transforms import transform_landmarks


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
        s = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", label)                    # np.float32(0.04) -> 0.04
        s = re.sub(r"array\((\[[^\]]*\]),\s*dtype=\w+\)", r"\1", s)     # array([a, b], dtype=float32) -> [a, b]
        arr =ast.literal_eval(s)
       
    
        
        coordinates = torch.tensor(arr, dtype=torch.float32)

        assert coordinates.shape == (4, 2), coordinates.shape
       

        if self.transform:
            image = self.transform(image)
 
        return image, coordinates

class BinaryDataset(Dataset):

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

      
        label= torch.tensor(label, dtype=torch.float32)
       
        if self.transform:
            image = self.transform(image)
 
        return image, label

full_dataset= DistanceDataset(
    annotations_file='data/dataset/labels/distance.csv',
    img_dir="data/dataset/images",
    transform= transform_landmarks

)

binary_dataset= BinaryDataset(
    annotations_file="data/dataset/labels/hand.csv",
    img_dir="data/dataset/images",
    transform=transform_landmarks

)

train_size = int(0.7 * len(full_dataset)) # 70% of all datasets are for training
test_size = len(full_dataset)- train_size# 15% for testing

bin_train_size=int(0.7* len(binary_dataset)) # 70%
bin_test_size=  len(binary_dataset) - bin_train_size # 15 %


training_dataset,test_dataset=random_split(full_dataset,[train_size,test_size]) # getting actual datasets from sizes
subset = torch.utils.data.Subset(training_dataset,range(32)) # subset for testing

bin_training_dataset, bin_test_dataset= random_split(binary_dataset,[bin_train_size,bin_test_size])
subset_bin = torch.utils.data.Subset(binary_dataset,range(32))

print(f"training datasets: {training_dataset.indices}")
print(f"test_dataset: {test_dataset.indices}")
