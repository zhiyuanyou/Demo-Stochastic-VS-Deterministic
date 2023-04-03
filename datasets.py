import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler


def build_dataloader(cfg):
    dataset = CustomDataset(cfg["meta_file"])
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=RandomSampler(dataset),
    )
    return data_loader


class CustomDataset(Dataset):
    def __init__(self, meta_file):
        self.meta_file = meta_file
        with open(meta_file, "r") as fr:
            self.data = []
            for line in fr:
                x1, x2 = line.strip().split()
                x1, x2 = float(x1), float(x2)
                self.data.append([x1, x2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x1, x2 = self.data[index]
        return torch.tensor([x1]), torch.tensor([x2])
