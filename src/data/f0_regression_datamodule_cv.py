from argparse import ArgumentError
from typing import Optional, Tuple
import pickle
import os, sys
import json

import torch

# import datasets
from pathlib import Path
from lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision.transforms import transforms
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import DataCollatorWithPadding
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, AutoModel
from src.utils.gpt2_letter_tokenizer import CustomGPT2Tokenizer, mGPTTokenizer, mBERTTokenizer, CustomBERTTokenizer

from src.data.components.feature_extractors import ProsodyFeatureExtractor
from src.data.components.datasets import TokenTaggingDataset
from src.data.components.collators import vector_collate_fn, encode_and_pad_batch


class F0RegressionDataModule(LightningDataModule):
    """
    LightningDataModule for HF Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        lab_root: str,
        phoneme_lab_root: str,
        wav_root: str,
        data_cache: str,
        train_file: str,
        val_file: str,
        test_file: str,
        dataset_name: str,
        use_fast_tokenizer: bool = False,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 1,
        pin_memory: bool = False,
        n_splits: int = 5,
        model_name: str = None,
        tokenization_by_letter: bool = False,
        f0_mode: str = "dct",
        f0_n_coeffs: int = 4,
        celex_path: str = None,
        stress_localizer: str = "celex",
        score_first_token: bool = False,
        score_last_token: bool = False,
        relative_to_prev: bool = False,
        n_prev: int = 1,
        relative_to_mean: bool = False,
        word_stats_path: str = None,
        debug: bool = False,
        explicit_words_length: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.tokenizer = None
        self.collator_fn = None
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.current_fold = 0

        self.keep_columns = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "bias",
            "teacher_probs",
        ]

    def prepare_data(self):
        """
        We should not assign anything here, so this function simply ensures
        that the pre-processed data is available.
        """
        pass

    def prepare_dataset(self, file_name):
        data_cache_path = os.path.join(self.hparams.data_cache, file_name)
        data_id = f"f0_{self.hparams.f0_mode}_{self.hparams.f0_n_coeffs}.pkl"

        if os.path.exists(data_cache_path) and data_id in os.listdir(data_cache_path):
            print(f"Loading data from cache: {data_cache_path, data_id}")
            # Load the data from cache
            with open(os.path.join(data_cache_path, data_id), "rb") as f:
                data = pickle.load(f)
            texts, f0_curves = data["texts"], data["f0"]
        else:
            print(f"Data not in cache: {data_cache_path, data_id}")
            # Data not in cache, create it
            extractor = ProsodyFeatureExtractor(
                lab_root=os.path.join(self.hparams.lab_root, file_name),
                wav_root=os.path.join(self.hparams.wav_root, file_name),
                phoneme_lab_root=os.path.join(self.hparams.phoneme_lab_root, file_name),
                data_cache=self.hparams.data_cache,
                extract_f0=True,
                f0_mode=self.hparams.f0_mode,
                f0_n_coeffs=self.hparams.f0_n_coeffs,
                celex_path=self.hparams.celex_path,
            )

            texts = extractor.get_all_text()
            f0_curves = extractor.get_all_f0()

            # Save the data to cache
            data = {"texts": texts, "f0": f0_curves}
            if not os.path.exists(data_cache_path):
                os.makedirs(data_cache_path)
            with open(os.path.join(data_cache_path, data_id), "wb") as f:
                pickle.dump(data, f)

            print(f"Saved data to cache: {data_cache_path, data_id}")

        dataset = TokenTaggingDataset(
            input_texts=texts,
            targets=f0_curves,
            tokenizer=self.tokenizer,
            model_name=self.hparams.model_name,
            score_first_token=self.hparams.score_first_token,
            score_last_token=self.hparams.score_last_token,
            relative_to_prev=self.hparams.relative_to_prev,
            n_prev=self.hparams.n_prev,
            relative_to_mean=self.hparams.relative_to_mean,
            word_stats=self.hparams.word_stats_path,
            debug=self.hparams.debug,
            explicit_words_length=self.hparams.explicit_words_length,
        )

        return texts, f0_curves, dataset

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        if not self.hparams.relative_to_mean:
            self.hparams.word_stats = None  # pass None to the dataset
        elif self.hparams.relative_to_mean and not self.hparams.word_stats_path:
            raise ValueError(
                "If relative_to_mean is True, you must provide a word_stats_path."
            )
        else:
            self.hparams.word_stats = json.load(open(self.hparams.word_stats_path, "r"))

        if not self.tokenizer:
            if "gpt2" in self.hparams.model_name:
                print("Using GPT2 tokenizer")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    self.hparams.model_name, add_prefix_space=True
                )
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif "bert" in self.hparams.model_name.lower():
                if self.hparams.tokenization_by_letter:
                    print('Using mBERT letter tokenizer')
                    self.tokenizer = mBERTTokenizer.from_pretrained(self.hparams.model_name)
                    # Use the correct padding token
                    # self.tokenizer.pad_token_id = self.tokenizer.sep_token_id
                    self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.convert_tokens_to_ids("[PAD]")
                else:
                    print(f"Using {self.hparams.model_name} tokenizer.")
                    ## uncomment the line below to use the custom tokenizer for filtering out misaligned tokens
                    # self.tokenizer = CustomBERTTokenizer.from_pretrained(self.hparams.model_name)
                    ## uncomment the line below to use the original tokenizer
                    self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model_name)
                    self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.convert_tokens_to_ids("[PAD]")
            elif "mGPT" in self.hparams.model_name:
                if self.hparams.tokenization_by_letter:
                    print('Using mGPT letter tokenizer')
                    self.tokenizer = mGPTTokenizer.from_pretrained('ai-forever/mGPT')
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                else:
                    print(f'Using {self.hparams.model_name} tokenizer')
                    # self.tokenizer = CustomGPT2Tokenizer.from_pretrained('ai-forever/mGPT')
                    self.tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/mGPT')
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif "llama" in self.hparams.model_name:
                if self.hparams.tokenization_by_letter:
                    print('Using Llama letter tokenizer')
                    self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                else:
                    print('Using Llama tokenizer')
                    self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
                    def tok(text):
                        words = text.split(" ")
                        return words
                    self.tokenizer.tokenize = tok
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            else:
                raise ValueError("Model name not recognized.")
        self.pad_token_id = self.tokenizer.pad_token_id
        print(f"Dataloader: padding with token id: {self.pad_token_id}")

        train_texts, train_durations, train_dataset = self.prepare_dataset(self.hparams.train_file)
        val_texts, val_durations, val_dataset = self.prepare_dataset(self.hparams.val_file)
        test_texts, test_durations, test_dataset = self.prepare_dataset(self.hparams.test_file)

        full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
        full_texts = train_texts + val_texts + test_texts
        full_durations = train_durations + val_durations + test_durations

        indices = torch.randperm(len(full_dataset)).tolist()
        # print("Full Dataset Length: ", len(full_dataset))

        test_size = len(full_dataset) // 5
        self.test_dataset = Subset(full_dataset, indices[-test_size:])
        print(f"Test dataset size (should be fixed): {len(self.test_dataset)}")
        train_val_indices = indices[:-test_size]
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(train_val_indices)):
            if fold == self.current_fold:
                self.train_dataset = Subset(full_dataset, [train_val_indices[i] for i in train_idx])
                self.val_dataset = Subset(full_dataset, [train_val_indices[i] for i in val_idx])
                break
        
        print(f"Fold {self.current_fold}: Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}, Test size: {len(self.test_dataset)}")

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def next_fold(self):
        self.current_fold = (self.current_fold + 1) % 5

    # def encode_collate(self, batch):
    #     return encode_and_pad_batch(batch, self.tokenizer, self.hparams.model_name)

    def collate(self, batch):
        return vector_collate_fn(batch, self.pad_token_id)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )
