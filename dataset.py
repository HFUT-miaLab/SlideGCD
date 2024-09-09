import csv
import os
import random
import joblib
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as pyg_Dataset


class FeatDataset(Dataset):
    def __init__(self, feature_root, slide_ids, labels):
        self.feature_root = feature_root
        self.slide_ids = slide_ids
        self.labels = labels

    def __getitem__(self, index):
        return (torch.from_numpy(torch.load(os.path.join(self.feature_root, self.slide_ids[index] + '.pth'))),
                self.labels[index])

    def __len__(self):
        return len(self.slide_ids)


class OriginDataset(Dataset):
    def __init__(self, feature_root, slide_ids, labels, num_instance=500):
        self.feature_root = feature_root
        self.slide_ids = slide_ids
        self.labels = labels
        self.num_instance = num_instance

    def __getitem__(self, index):
        feats = torch.load(os.path.join(self.feature_root, self.slide_ids[index] + '.pth'))['feature']

        if feats.shape[0] > self.num_instance:
            sample_idx = random.sample(range(feats.shape[0]), k=self.num_instance)
            return torch.from_numpy(feats[sample_idx]), self.labels[index]
        else:
            instances = torch.zeros(size=(self.num_instance, feats.shape[1]))
            instances[:feats.shape[0]] = torch.from_numpy(feats)
            return instances, self.labels[index]

    def __len__(self):
        return len(self.slide_ids)


class PatchGCNDataset(Dataset):
    def __init__(self, graph_root, names, labels):
        self.graph_paths = [os.path.join(graph_root, name + '.pkl') for name in names]
        self.labels = labels
        assert len(self.graph_paths) == len(self.labels)

    def __getitem__(self, index):
        G = joblib.load(self.graph_paths[index])
        label = self.labels[index]

        return Data(x=G.x, edge_index=G.edge_index, edge_latent=G.edge_latent, y=label)

    def __len__(self):
        return len(self.labels)


def load_trainval(csvfile, fold):
    ids_train, labels_train, ids_val, labels_val = [], [], [], []

    with open(csvfile, 'r') as file:
        reader = csv.reader(file)
        for idx, content in enumerate(reader):
            slide_id = content[1]
            label = int(content[2])
            fold_idx = int(content[-1])

            if fold_idx == fold:
                ids_val.append(slide_id)
                labels_val.append(label)
            else:
                ids_train.append(slide_id)
                labels_train.append(label)

    _, cls_count = np.unique(np.array(labels_train), return_counts=True)
    cls_weights = np.sum(cls_count) / cls_count
    train_weights = [cls_weights[label] for label in labels_train]

    return ids_train, labels_train, train_weights, ids_val, labels_val


def load_test(csvfile):
    ids, labels = [], []

    with open(csvfile, 'r') as file:
        reader = csv.reader(file)
        for idx, content in enumerate(reader):
            slide_id = content[1]
            label = int(content[2])

            ids.append(slide_id)
            labels.append(label)

    return ids, labels