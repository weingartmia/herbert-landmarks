
import sys
import data.distance_datasets as distance_datasets
from torch.utils.data import DataLoader

sys.path.append("..")
from utils.loop_dataloader import show_index

train_loader = DataLoader(distance_datasets.training_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(distance_datasets.test_dataset, batch_size=32, shuffle=False)

subset = DataLoader(distance_datasets.subset,batch_size=32,shuffle=True)

bin_train_loader = DataLoader(distance_datasets.bin_training_dataset,batch_size=32, shuffle=True)
bin_test_loader= DataLoader(distance_datasets.bin_test_dataset,batch_size=32,shuffle=False)

subset_bin = DataLoader(distance_datasets.subset_bin,batch_size=32,shuffle=True)
# show_index(train_loader,0)