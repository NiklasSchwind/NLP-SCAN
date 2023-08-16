from sentence_transformers import SentenceTransformer
import torch
from typing import Union, Literal, List
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel


class Embedder:
    def __init__(self, embedding_method: Literal['IndicativeSentence', 'SBert'],
                 indicative_sentence: str = None,
                 indicative_sentence_position: Literal['first', 'last'] = None,
                 device: str = 'cpu',
                ):

        self.embedding_method = embedding_method
        self.indicative_sentence = indicative_sentence
        self.indicative_sentence_position = indicative_sentence_position
        self.device = device

    def embed_texts_indicative_sentence(self, texts: List[str]):

        embedding_text = []
        model_name = 'roberta-base'
        for text in texts:
            if self.indicative_sentence_position == 'first':
                embedding_text.append(self.indicative_sentence + text)
            elif self.indicative_sentence_position == 'last':
                embedding_text.append(text + self.indicative_sentence)

        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name).to(self.device)

        num_sentences = len(texts)
        num_batches = (num_sentences + 64 - 1) // 64

        # Initialize a list to store the mask token encodings for all batches
        mask_token_encodings = []

        # Process each batch of input sentences
        for i in range(num_batches):
            start = i * 64
            end = min((i + 1) * 64, num_sentences)
            # Extract the input tensors for the current batch
            input_ids = []
            attention_mask = []
            for text in embedding_text[start:end]:
                encoded_inputs = tokenizer.encode_plus(text, padding='max_length', return_tensors='pt').to(
                    self.device)
                if self.indicative_sentence_position == 'first':
                    input_ids.append(encoded_inputs['input_ids'][0, :512])
                    attention_mask.append(encoded_inputs['attention_mask'][0, :512])
                elif self.indicative_sentence_position == 'last':
                    input_ids.append(encoded_inputs['input_ids'][0, -512:])
                    attention_mask.append(encoded_inputs['attention_mask'][0, -512:])
            input_ids = torch.cat(input_ids, dim=0).reshape((end - start, 512))
            attention_mask = torch.cat(attention_mask, dim=0).reshape((end - start, 512))
            # Feed the input tensors to the RoBERTa model
            with torch.no_grad():
                batch_output = model(input_ids, attention_mask=attention_mask)  # , attention_mask=attention_mask)
            # Retrieve the encodings of the mask tokens from the output tensor
            mask_token_indices = torch.where(input_ids == tokenizer.mask_token_id)
            batch_mask_token_encodings = batch_output[0][mask_token_indices[0], mask_token_indices[1], :]
            # Add the mask token encodings for the current batch to the list
            mask_token_encodings.append(batch_mask_token_encodings)

        return torch.cat(mask_token_encodings, dim=0).to(self.device)

    def embed_texts_SBert(self, texts: List[str]):

        embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=self.device)
        embedder.max_seq_length = 128
        corpus_embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=False)

        return torch.from_numpy(corpus_embeddings).to(self.device)

    def embed_texts(self, texts: List[str]):

        if self.embedding_method == 'IndicativeSentence':

            if self.indicative_sentence_position in ['first', 'last'] and '<mask>' in self.indicative_sentence:
                self.embed_texts_indicative_sentence(texts)
            elif self.indicative_sentence_position not in ['first', 'last']:
                raise ValueError(
                    f"Invalid Indicative Sentence Position: '{self.indicative_sentence_position}' is not one of the valid options. You can only select \"last\" and \"first\".")
            elif not isinstance(self.indicative_sentence, str):
                raise ValueError(
                    f"Invalid Indicative Sentence: '{self.indicative_sentence}' has to be a string.")
            elif not '<mask>' in self.indicative_sentence:
                raise ValueError(
                    f"Invalid Indicative Sentence: '{self.indicative_sentence}' doesn't include <mask> token!")
        elif self.embedding_method == 'SBert':
            self.embed_texts_SBert(texts)
        else:
            raise ValueError(
                f"Invalid Embedding Method: '{self.embedding_method}' is not one of the valid options. You can only select \"IndicativeSentence\" and \"SBert\".")
