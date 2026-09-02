import sys
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.io import decode_image

sys.path.append("..")
from transforms.transforms import transform

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

full_dataset= DistanceDataset(
    annotations_file='data/dataset/labels.csv',
    img_dir="data/dataset/images",
    transform= transform

)

train_size = int(0.7 * len(full_dataset)) # 70% of all datasets are for training
val_size = int(0.15 * len(full_dataset)) # 15% for 
test_size = len(full_dataset) - train_size - val_size

training_dataset,val_dataset,test_dataset=random_split(full_dataset,[train_size,val_size,test_size])
print(f"training datasets: {training_dataset.indices}")
print(f"val_dataset: {val_dataset.indices}")
print(f"test_dataset: {test_dataset.indices}")
# training_data = DistanceDataset(
#     annotations_file='dataset/labels.csv',
#     img_dir="dataset/images",
#     transform= transform


# )

# test_data = DistanceDataset(
#    annotations_file='dataset/labels.csv',
#    img_dir='dataset/images/',
#    transform=transform
# )