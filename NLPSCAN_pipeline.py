from typing import Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pathlib import Path
import torch
import pandas as pd
from Embedder import Embedder
from NeighborDataset import Neighbor_Dataset
from NLPSCAN_Trainer import NLPSCAN_Trainer, DocScanDataset
from FineTuningThroughSelfLabeling import FinetuningThroughSelflabeling

class NLPSCAN:

    def __init__(self,
                 file_path_data: Union[str, Path],
                 file_path_result: Union[str, Path],
                 embedding_method: Literal['IndicativeSentence', 'SBert'],
                 num_classes: int,
                 indicative_sentence: str = None,
                 indicative_sentence_position: Literal['first', 'last'] = None,
                 device: str = 'cpu',
                 num_neighbors: int = 5,
                 batch_size: int = 64,
                 entropy_weight: float = None,
                 dropout: float = 0.1,
                 clustering_method: str = None,
                 num_epochs: int = 5,
                 prototype_threshold: float = 0.95,
                ):
        self.file_path_data = file_path_data
        self.file_path_result = file_path_result
        self.embedding_method = embedding_method
        self.indicative_sentence = indicative_sentence
        self.indicative_sentence_position = indicative_sentence_position
        self.device = device
        self.num_neighbors = num_neighbors
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.prototype_threshold = prototype_threshold
        self.Trainer = None

        if entropy_weight is None and embedding_method == 'IndicativeSentence':
            self.entropy_weight = 3
        elif entropy_weight is None and embedding_method == 'SBert':
            self.entropy_weight = 2
        else:
            self.entropy_weight = entropy_weight

        if clustering_method is None and embedding_method == 'IndicativeSentence':
            self.clustering_method = 'EntropyLoss'
        elif clustering_method is None and embedding_method == 'SBert':
            self.clustering_method = 'SCANLoss'
        else:
            self.clustering_method = clustering_method


    def load_data(self, path:Union[str, Path]):
        with open(path) as f:
            sentences = [line.strip() for line in f]
        df_texts = pd.DataFrame(sentences, columns=["sentence"])
        return df_texts



    # Train NLP-SCAN with dataset and afterwards classify same dataset
    def TrainAndClassify(self):

        print("Loading Data...")

        df_texts = self.load_data(self.file_path_data)

        print("Embedding Sentences...")

        embedder = Embedder(embedding_method = self.embedding_method,
                 indicative_sentence = self.indicative_sentence,
                 indicative_sentence_position = self.indicative_sentence_position,
                 device=self.device)

        text_embedded = embedder.embed_texts(df_texts['sentence'])

        print(text_embedded.shape)

        print("Retrieving Neighbors...")

        NeighborDataset = Neighbor_Dataset(num_neighbors= self.num_neighbors, num_classes = self.num_classes, device = self.device)

        neighbor_dataset = NeighborDataset.create_neighbor_dataset(text_embedded)

        print("Training Classification Model...")

        Trainer = NLPSCAN_Trainer(num_classes=self.num_classes, device=self.device, dropout=self.dropout,
                                  batch_size=self.batch_size, hidden_dim=len(text_embedded[-1]),
                                  method=self.clustering_method)

        Trainer.train_model(neighbor_dataset=neighbor_dataset, train_dataset_embeddings=text_embedded,
                            num_epochs=self.num_epochs, entropy_weight=self.entropy_weight)

        print("Refining Classification Model...")

        SelfLabeling = FinetuningThroughSelflabeling(model_trainer=Trainer,
                                                     embedder=embedder, train_data=df_texts, train_embeddings=text_embedded,
                                                     neighbor_dataset=neighbor_dataset,
                                                     batch_size=self.batch_size, device=self.device,
                                                     threshold=self.prototype_threshold,
                                                     clustering_method=self.clustering_method)

        num_prototypes_before = SelfLabeling.num_prototypes
        num_prototypes = SelfLabeling.num_prototypes + 1

        while num_prototypes_before < num_prototypes:
            num_prototypes_before = num_prototypes
            SelfLabeling.fine_tune_through_selflabeling_fast()
            num_prototypes = SelfLabeling.num_prototypes

        print("Writing Classification Result...")

        predict_dataset = DocScanDataset(neighbor_dataset, text_embedded, mode="predict",
                                               test_embeddings=text_embedded, device=self.device,
                                               method=self.clustering_method)

        predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                         collate_fn=predict_dataset.collate_fn_predict,
                                                         batch_size=self.batch_size)

        predictions, probabilities = Trainer.get_predictions(predict_dataloader)

        df_texts['class'] = predictions

        df_texts.to_csv(self.file_path_result, index=False)

    # Train NLP-SCAN
    def Train(self):

        print("Loading Data...")

        df_texts = self.load_data(self.file_path_data)

        print("Embedding Sentences...")

        embedder = Embedder(embedding_method=self.embedding_method,
                            indicative_sentence=self.indicative_sentence,
                            indicative_sentence_position=self.indicative_sentence_position,
                            device=self.device)

        text_embedded = embedder.embed_texts(df_texts['sentence'])

        print("Retrieving Neighbors...")

        NeighborDataset = Neighbor_Dataset(num_neighbors=self.num_neighbors, num_classes=self.num_classes,
                                           device=self.device)

        neighbor_dataset = NeighborDataset.create_neighbor_dataset(text_embedded)

        print("Training Classification Model...")

        Trainer = NLPSCAN_Trainer(num_classes=self.num_classes, device=self.device, dropout=self.dropout,
                                  batch_size=self.batch_size, hidden_dim=len(text_embedded[-1]),
                                  method=self.clustering_method)

        Trainer.train_model(neighbor_dataset=neighbor_dataset, train_dataset_embeddings=text_embedded,
                            num_epochs=self.num_epochs, entropy_weight=self.entropy_weight)

        print("Refining Classification Model...")

        SelfLabeling = FinetuningThroughSelflabeling(model_trainer=Trainer,
                                                     embedder=embedder, train_data=df_texts,
                                                     train_embeddings=text_embedded,
                                                     neighbor_dataset=neighbor_dataset,
                                                     batch_size=self.batch_size, device=self.device,
                                                     threshold=self.prototype_threshold,
                                                     clustering_method=self.clustering_method)

        num_prototypes_before = SelfLabeling.num_prototypes
        num_prototypes = SelfLabeling.num_prototypes + 1

        while num_prototypes_before < num_prototypes:
            num_prototypes_before = num_prototypes
            SelfLabeling.fine_tune_through_selflabeling_fast()
            num_prototypes = SelfLabeling.num_prototypes

        self.Trainer = Trainer

    # Classify dataset with already trained NLP-SCAN
    def Classify(self, unsupervised_dataset_path:  Union[str, Path], result_path:  Union[str, Path]):

        if self.Trainer is None:
            raise ValueError(f"Trying to classify before Training: Please train NLP-SCAN first.")

        print("Loading Data...")

        df_texts = self.load_data(unsupervised_dataset_path)

        print("Embedding Sentences...")

        embedder = Embedder(embedding_method=self.embedding_method,
                            indicative_sentence=self.indicative_sentence,
                            indicative_sentence_position=self.indicative_sentence_position,
                            device=self.device)

        text_embedded = embedder.embed_texts(df_texts['sentence'])

        print("Retrieving Neighbors...")

        NeighborDataset = Neighbor_Dataset(num_neighbors=self.num_neighbors, num_classes=self.num_classes,
                                           device=self.device)

        neighbor_dataset = NeighborDataset.create_neighbor_dataset(text_embedded)

        print("Writing Classification Result...")

        predict_dataset = DocScanDataset(neighbor_dataset, text_embedded, mode="predict",
                                         test_embeddings=text_embedded, device=self.device,
                                         method=self.clustering_method)

        predict_dataloader = torch.utils.data.DataLoader(predict_dataset, shuffle=False,
                                                         collate_fn=predict_dataset.collate_fn_predict,
                                                         batch_size=self.batch_size)

        predictions, probabilities = self.Trainer.get_predictions(predict_dataloader)

        df_texts['class'] = predictions

        df_texts.to_csv(result_path, index=False)



