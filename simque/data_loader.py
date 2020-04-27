import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class data_set(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        labels = torch.tensor([item[0] for item in data])
        neg_labels = [torch.tensor(item[1]) for item in data]
        sentences = [torch.tensor(item[2]) for item in data]
        lenghts = [torch.tensor(item[3]) for item in data]
        return (
            labels,
            neg_labels,
            sentences,
            lenghts
        )

def get_data_loader(config, data, shuffle = True, drop_last = False, batch_size = None):
    dataset = data_set(data)
    if batch_size == None:
        batch_size = min(config['batch_size_per_step'], len(data))
    else:
        batch_size = min(batch_size, len(data))
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        pin_memory = True,
        num_workers = config['num_workers'],
        collate_fn = dataset.collate_fn,
        drop_last = drop_last)
    return data_loader