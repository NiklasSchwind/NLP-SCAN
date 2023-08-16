from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F



def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=1e-8)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight=2.0, entropy="entropy", experiment=None):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0
        self.entropy = entropy
        # if target_probs is not None:
        #    self.target_probs = target_probs

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        # print (consistency_loss, entropy_loss)
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)

class ConfidenceBasedCE(nn.Module):
    def __init__(self, device, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing
        self.device = device
        self.to(self.device)

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling
        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts=True)
            freq = 1 / (counts.float() / n)

            weight = torch.ones(c).to(self.device)

            weight[idx] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight=weight, reduction='mean')

        return loss

class DocScanModel(torch.nn.Module):
    def __init__(self, num_labels, dropout, hidden_dim=768, device = 'cpu'):
        super(DocScanModel, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)
        self.device = device
        self.dropout = dropout

    def forward(self, feature):
        if self.dropout is not None:
            dropout = torch.nn.Dropout(p=self.dropout)
            feature = dropout(feature)
        output = self.classifier(feature)
        return output



class DocScanDataset(torch.utils.data.Dataset):
    def __init__(self, neighbor_df, embeddings, test_embeddings="", mode="train", device = 'cpu', method = 'SCANLoss' ):
        self.neighbor_df = neighbor_df
        self.embeddings = embeddings
        self.mode = mode
        self.device = device
        self.method = method
        if mode == "train":
            self.examples = self.load_data()
        elif mode == "predict":
            self.examples = test_embeddings

    def load_data(self):
        examples = []
        for i ,j in zip(self.neighbor_df["anchor"], self.neighbor_df["neighbor"]):
            examples.append((i ,j))
        random.shuffle(examples)
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.mode == "train":
            anchor, neighbor = self.examples[item]
            sample = {"anchor": anchor, "neighbor": neighbor}
        elif self.mode == "predict":
            anchor = self.examples[item]
            sample = {"anchor": anchor}
        return sample
    def collate_fn(self, batch):

        anchors = torch.tensor([i["anchor"] for i in batch])
        out = self.embeddings[anchors].to(self.device)
        if self.method == 'SCANLoss':
            neighbors = torch.tensor([i["neighbor"] for i in batch])
        elif self.method == 'EntropyLoss':
            neighbors = torch.tensor([i["anchor"] for i in batch])
        out_2 = self.embeddings[neighbors].to(self.device)
        return {"anchor": out, "neighbor": out_2}

    def collate_fn_predict(self, batch):
        out = torch.vstack([i["anchor"] for i in batch]).to(self.device)
        return {"anchor": out}


class NLPSCAN_Trainer:
    def __init__(self, num_classes, device, dropout, batch_size, hidden_dim, method):

        self.model = DocScanModel(num_labels=num_classes, dropout=dropout, hidden_dim=hidden_dim, device = device).to(device)
        self.device = device
        self.num_classes = num_classes
        self.dropout = dropout
        self.batch_size = batch_size
        self.method = method

    def get_predictions(self, dataloader):
        predictions, probs = [], []
        epoch_iterator = tqdm(dataloader, total=len(dataloader))
        self.model.eval()
        print(len(dataloader))
        with torch.no_grad():
            for i, batch in enumerate(epoch_iterator):
                self.model.eval()
                output_i = self.model(batch["anchor"])
                # probs.extend(torch.nn.functional.softmax(output_i, dim=1).cpu().numpy())
                probs.extend(output_i.cpu().numpy())
                predictions.extend(torch.argmax(output_i, dim=1).cpu().numpy())
        print(len(predictions))
        return predictions, probs

    def train(self, optimizer, criterion, train_dataloader, num_epochs):
        train_iterator = range(int(num_epochs))
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        softmax = torch.nn.Softmax()
        # train

        #targets_map = {i: j for j, i in enumerate(np.unique(self.df_test["label"]))}
        #targets = [targets_map[i] for i in self.df_test["label"]]

        for epoch in train_iterator:
            bar_desc = "Epoch %d of %d | num classes %d | Iteration" % (epoch + 1, len(train_iterator), self.num_classes)
            epoch_iterator = tqdm(train_dataloader, desc=bar_desc)
            for step, batch in enumerate(epoch_iterator):
                anchor, neighbor = batch["anchor"], batch["neighbor"]

                anchors_output, neighbors_output = self.model(anchor), self.model(neighbor)

                total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.model.zero_grad()
        # predictions, probabilities = self.get_predictions(model, predict_dataloader)
        # evaluate(np.array(targets), np.array(predictions),verbose=0)

        optimizer.zero_grad()
        self.model.zero_grad()


    def train_model(self, neighbor_dataset, train_dataset_embeddings, num_epochs, entropy_weight = 2.0):
        train_dataset = DocScanDataset(neighbor_dataset, train_dataset_embeddings, mode="train", device = self.device, method = self.method)
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = SCANLoss(entropy_weight = entropy_weight)
        criterion.to(self.device)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=train_dataset.collate_fn,
                                                       batch_size=self.batch_size)
        # train
        self.model.train()
        self.train(optimizer, criterion, train_dataloader, num_epochs)

    def give_model(self):

        return self.model

    def train_selflabeling(self, prototype_embeddings, augmented_prototype_embeddings, threshold = 0.99, num_epochs = 5,augmentation_method = '' ):

        self.model.to(self.device)
        if augmentation_method == 'Dropout':
            self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = ConfidenceBasedCE(device = self.device,threshold=threshold, apply_class_balancing=True)


        dataset = list(zip(prototype_embeddings, augmented_prototype_embeddings))

        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        train_iterator = range(int(num_epochs))

        for epoch in train_iterator:
            bar_desc = "Epoch %d of %d | num classes %d | Iteration" % (
                epoch + 1, len(train_iterator), self.num_classes)

            epoch_iterator = tqdm(dataloader, desc=bar_desc)
            for step, batch in enumerate(epoch_iterator):
                try:
                    anchor_weak, anchor_strong = batch[0].to(self.device), batch[1].to(self.device)
                    original_output, augmented_output = self.model(anchor_weak), self.model(anchor_strong)
                    total_loss = criterion(original_output, augmented_output)
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    self.model.zero_grad()
                except ValueError:
                    print(f'Recieved Value Error in step {step}')
        optimizer.zero_grad()
        self.model.zero_grad()
