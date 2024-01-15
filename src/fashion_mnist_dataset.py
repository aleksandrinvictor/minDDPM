from typing import Callable, Optional

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import FashionMNIST


class DDPMFashionMNIST(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        self.dataset = FashionMNIST(root=root, train=train, transform=transform, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, label = self.dataset[index]

        return image
