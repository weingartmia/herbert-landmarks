
import sys
import data.distance_datasets as distance_datasets
from torch.utils.data import DataLoader

sys.path.append("..")
from utils.loop_dataloader import show_index

train_loader = DataLoader(distance_datasets.training_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(distance_datasets.val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(distance_datasets.test_dataset, batch_size=32, shuffle=False)

print(f"len of train data: {train_loader}")
print(f"len of val loader: {len(val_loader)}")
print(f"len of test loader: {len(test_loader)}")

show_index(train_loader,0)