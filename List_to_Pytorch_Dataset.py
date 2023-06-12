import torch
import torchvision
from torch.utils.data import Dataset
import CSV_to_List
import torchvision.transforms as transforms

labels = {"STD" : 0,
          "WAL" : 1,
          "JOG" : 2,
          "JUM" : 3,
          "STU" : 4,
          "STN" : 5,
          "SCH" : 6,
          "SIT" : 7,
          "CHU" : 8,
          "CSI" : 9,
          "CSO" : 10,
          "FOL" : 11,
          "FKL" : 12,
          "BSC" : 13,
          "SDL" : 14
          #"PFF" : 15
          }
          
class MobifallData(Dataset):
    def __init__(self, augment):
        self.samples = CSV_to_List.get_dataset(augment = augment, reduced_frequency=True, target_frequency_ratio=4)


    def __len__(self):
        return len(self.samples)

    #returns data, label
    def __getitem__(self, idx):
         data = [torch.FloatTensor(channel) for channel in self.samples[idx][0]]
         target = labels[self.samples[idx][1]]

         return torch.stack(data), target
