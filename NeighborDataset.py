import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import torch
from sklearn import metrics

def evaluate(y, preds):
	print(metrics.classification_report(y, preds))
	#print(metrics.confusion_matrix(y, preds))
	print("accuracy", metrics.accuracy_score(y, preds))

class MemoryBank(object):
    def __init__(self, features, targets, n, dim, num_classes):
        self.n = n
        self.dim = dim
        # self.features = torch.FloatTensor(self.n, self.dim)
        # self.targets = torch.LongTensor(self.n)
        self.features = features
        self.targets = targets
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.C = num_classes

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True, show_eval=False):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        # index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk + 1)  # Sample itself is included

        # evaluate
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            if show_eval:
                print(np.shape(neighbor_targets), np.shape(anchor_targets))
                evaluate(anchor_targets.flatten(), neighbor_targets.flatten())
            return indices, accuracy

        else:
            return indices

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


class Neighbor_Dataset:

    def __init__(self, num_neighbors: int, num_classes: int, device: str):

        self.num_neighbors = num_neighbors
        self.num_classes = num_classes
        self.device = device

    def _create_neighbor_dataset(self,  memory_bank = None, indices=None,safeNeighborDataset = False):
        if indices is None:
            indices = memory_bank.mine_nearest_neighbors(self.num_neighbors, show_eval=False,
                                                         calculate_accuracy=False)
        examples = []
        for i, index in enumerate(indices):
            anchor = i
            neighbors = index
            for neighbor in neighbors:
                if neighbor == i:
                    continue
                examples.append((anchor, neighbor))
        df = pd.DataFrame(examples, columns=["anchor", "neighbor"])
        return df

    def _retrieve_neighbours_gpu(self, X, batchsize=16384, num_neighbors=5):
        #This is ok since it is not an exact algorithm :)
        import faiss
        res = faiss.StandardGpuResources()  # use a single GPU
        n, dim = X.shape[0], X.shape[1]
        index = faiss.IndexFlatIP(dim)  # create CPU index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)  # create GPU index
        gpu_index_flat.add(X)  # add vectors to the index

        all_indices = []
        for i in tqdm(range(0, n, batchsize)):
            features = X[i:i + batchsize]
            distances, indices = gpu_index_flat.search(features, num_neighbors)
            all_indices.extend(indices)
        return all_indices

    def create_neighbor_dataset(self, data):

        if self.device == "cpu":
            memory_bank = MemoryBank(data, "", len(data),
                                          data.shape[-1],
                                          self.num_classes)
            neighbor_dataset = self._create_neighbor_dataset(memory_bank = memory_bank)
        else:
            indices = self._retrieve_neighbours_gpu(data.cpu().numpy(), num_neighbors=self.num_neighbors)
            neighbor_dataset = self._create_neighbor_dataset(indices = indices)

        return neighbor_dataset