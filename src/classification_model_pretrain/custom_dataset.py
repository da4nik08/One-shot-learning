from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, images, labels, mode, transform):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.labels = labels
        self.images = images

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.images)

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        x = self.transform(self.images[index])
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            return x, torch.tensor(label, dtype=torch.long)