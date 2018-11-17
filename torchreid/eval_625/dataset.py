import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class evalDataset(Dataset):
    def __init__(self, path, transform):
        self.dir = path
        self.image = [f for f in os.listdir(self.dir) if f.endswith('png')]
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        name = self.image[idx]

        img = Image.open(os.path.join(self.dir, name))
        img = self.transform(img)

        return {'name': name.replace('.png', ''), 'img': img}
