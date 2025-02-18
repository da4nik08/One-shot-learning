import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random


class TripletDataset(Dataset):
    def __init__(self, images, labels, mode, transform, DATA_MODES):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.labels = labels            # np.array
        self.images = images            #

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.used_ind = {key: [] for key in np.unique(self.labels)}
        self.len_ = len(self.images)

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        img1 = self.transform(self.images[index])
        label = self.labels[index]
        
        indices = np.where(self.labels == label)[0]                   # all indices class
        if len(self.used_ind[label]) == len(indices):
            self.used_ind[label] = []
        indices = np.delete(indices, np.where(indices == index))      # delete ancor index
        if len(self.used_ind[label]) == len(indices) and index not in self.used_ind[label]:
            self.used_ind[label] = []  # !!!
        else:
            indices = np.delete(indices, np.where(np.isin(indices, self.used_ind[label])))    # delete used element as positive

        positive_idx = random.choice(indices)
        img2 = self.transform(self.images[positive_idx])
 
        self.used_ind[label].append(positive_idx)
        if self.mode == 'test':
            return img1, img2
        else:
            label = self.labels[index]
            return img1, img2, torch.tensor(label, dtype=torch.long)