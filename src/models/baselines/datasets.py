from torch.utils.data import Dataset, DataLoader
import pickle
import os
import torch
from fastText import FastTextModel
from GloVe import GloVeModel
from transformers import GPT2Tokenizer



class EmbeddingDataset(Dataset):
    def __init__(self, dir_path, mode, word_model, dataset_size=None):
        self.features = []
        self.labels = []
        self.mode = mode
        self.word_model = word_model
        '''        if isinstance(self.word_model, FastTextModel) == False and isinstance(self.word_model, GloVeModel) == False:
            self.tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/mGPT")
        '''
        if dataset_size:
            print(f"loading {dataset_size} samples from {mode} set")
            self.features = self.features[:dataset_size]
            self.labels = self.labels[:dataset_size]

        with open(os.path.join(dir_path, mode, f"f0_dct_4.pkl"), "rb") as f:
            self.data = pickle.load(f)
        '''with open(os.path.join(dir_path, f"{mode}_words.pkl"), "rb") as f:
            self.features = pickle.load(f)
        with open(os.path.join(dir_path, f"{mode}_labels.pkl"), "rb") as f:
            self.labels = pickle.load(f)'''
        
        #self.features = self.data["texts"]
        self.features = [word for text in self.data["texts"] for word in text.split()]

        #self.labels = self.data["f0"]
        self.labels = [item for sublist in self.data["f0"] for item in sublist]

        # convert to float 32 tensors
        self.labels = torch.tensor(self.labels).float()

        print(
            f"shapes of features and labels: {len(self.features)}, {self.labels.shape}"
        )

    def __len__(self):
        return len(self.features)

    # def __getitem__(self, idx):
    #     print(f"get item: {self.features[idx]}, {self.labels[idx]}")
    #     if isinstance(self.word_model, FastTextModel):
    #         return self.word_model.get_word_embedding(self.features[idx]), self.labels[idx]
    #     elif isinstance(self.word_model, GloVeModel):
    #         return self.word_model.get_word_embedding(self.features[idx]), self.labels[idx]
    #     '''else:
    #         indices = self.tokenizer(self.features[idx]).input_ids
    #         flat_indices = [index[0] for index in indices]
    #         return self.word_model[flat_indices,:], self.labels[idx]'''
    def __getitem__(self, idx):
        try:
            feature = self.features[idx]
            label = self.labels[idx]
            if isinstance(self.word_model, FastTextModel) or isinstance(self.word_model, GloVeModel):
                embedding = self.word_model.get_word_embedding(feature)
                return embedding, label
        except IndexError:
            # Fallback to a safe index or handling
            print(f"Index {idx} out of range for dataset size {len(self.labels)}.")
            last_idx = len(self.labels) - 1
            return self[last_idx]  # Recursively return the last item
