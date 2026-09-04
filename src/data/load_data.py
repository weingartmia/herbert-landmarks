
import sys
import data.distance_datasets as distance_datasets
from torch.utils.data import DataLoader

sys.path.append("..")
from utils.loop_dataloader import show_index

train_loader = DataLoader(distance_datasets.training_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(distance_datasets.test_dataset, batch_size=32, shuffle=False)

rot_train_loader = DataLoader(distance_datasets.rot_training_dataset,batch_size=32, shuffle=True)
rot_test_loader= DataLoader(distance_datasets.rot_test_dataset,batch_size=32,shuffle=False)


show_index(train_loader,0)