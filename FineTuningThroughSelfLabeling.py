import pandas as pd
import torch
from NLPSCAN_Trainer import DocScanDataset, DocSCAN_Trainer
import numpy as np
from Embedder import Embedder
from scipy.special import softmax
import copy


class FinetuningThroughSelflabeling:

    def __init__(self,
                 model_trainer: DocSCAN_Trainer,
                 embedder: Embedder,
                 train_data: pd.DataFrame,
                 train_embeddings: torch.tensor,
                 neighbor_dataset: pd.DataFrame,
                 batch_size: int,
                 device: str,
                 threshold: float,
                 clustering_method: str,
                 ):
        self.device = device
        self.model_trainer = model_trainer
        self.train_data = train_data
        self.batch_size = batch_size
        self.train_embeddings = train_embeddings
        self.neighbor_dataset = neighbor_dataset
        self.embedder = embedder
        self.threshold = threshold
        self.clustering_method = clustering_method
        self.num_prototypes = 0

    def mine_prototype_indexes(self, predict_dataset: DocScanDataset):

        predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                               collate_fn=predict_dataset.collate_fn_predict,
                                                               batch_size=self.batch_size)

        predictions_train, probabilities_train = self.model_trainer.get_predictions(predict_dataloader)

        probabilities_array = np.array(probabilities_train.tolist())

        softmax_probabilities = np.apply_along_axis(softmax, 1, probabilities_array)

        max_probabilities = np.max(softmax_probabilities, axis=1)

        indices = np.where(max_probabilities >= self.threshold)[0]

        return indices

    def fine_tune_through_selflabeling_fast(self):

        # train data
        predict_dataset_train = DocScanDataset(self.neighbor_dataset, self.train_embeddings, mode="predict",
                                               test_embeddings=self.train_embeddings, device=self.device,method = self.clustering_method)

        prototype_indexes = self.mine_prototype_indexes(predict_dataset_train)
        if len(prototype_indexes) == 0:
            self.num_prototypes = len(prototype_indexes)
        else:
            self.num_prototypes = len(prototype_indexes)
            print(f'Num Prototypes {self.num_prototypes}')
            prototypes = copy.deepcopy(self.train_embeddings[prototype_indexes])
            copy_prototypes  = copy.deepcopy(prototypes)


            self.model_trainer.train_selflabeling(prototypes, copy_prototypes, threshold = self.threshold, num_epochs = 5, augmentation_method = 'Dropout')



    def get_predictions(self, test_data):

        predictions, probabilities = self.model_trainer.get_predictions(test_data)

        return predictions, probabilities




