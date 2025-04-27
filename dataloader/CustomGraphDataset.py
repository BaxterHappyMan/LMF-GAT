import os.path
from torch_geometric.data import Dataset, Data
import torch
import numpy as np


class CustomGraphDataset(Dataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.name = name
        self.data = np.load(os.path.join(root, name), allow_pickle=True)

    @property
    def raw_file_names(self):
        return self.name

    @property
    def processed_file_names(self):
        return []

    def len(self) -> int:
        return len(self.data.files)

    def get(self, idx):
        graph = self.data[f"graph{idx}"].item()
        x = torch.tensor(graph["x"], dtype=torch.float)
        edge_index = torch.tensor(graph["adj"], dtype=torch.long)
        y = torch.tensor(graph["y"], dtype=torch.long)
        node_y = torch.tensor(graph["node_y"], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y, node_y = node_y)
