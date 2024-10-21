from argparse import ArgumentError
from typing import Optional, Tuple
import pickle
import os, sys
import json

import torch

# import datasets
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, AutoModel, TextDataset
from tokenizers import Tokenizer
from src.utils.gpt2_letter_tokenizer import CustomGPT2Tokenizer, mGPTTokenizer
from datasets import load_dataset

from src.data.components.feature_extractors import ProsodyFeatureExtractor
from src.data.components.datasets import TokenTaggingDataset, SubtitlesDataset
from src.data.components.collators import collate_fn, encode_and_pad_batch, collate_batch, vector_collate_fn


class NoDiacriticsDataModule(LightningDataModule):
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
        dataset_path: str,
        data_cache: str,
        remove_diacritics_path: str,
        train_file: str,
        val_file: str,
        test_file: str,
        dataset_name: str,
        dataset_size: int = 1000,
        use_fast_tokenizer: bool = False,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        model_name: str = None,
        score_first_token: bool = False,
        score_last_token: bool = False,
        relative_to_prev: bool = False,
        n_prev: int = 1,
        relative_to_mean: bool = False,
        word_stats_path: str = None,
        debug: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.tokenizer = None
        self.collator_fn = None

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
        data_id = "vietnamese_subtitles_notones.txt"
        if os.path.exists(data_cache_path) and data_id in os.listdir(data_cache_path):
            print(f"Loading data from cache: {data_cache_path, data_id}")
            # Load the data from cache
            with open(os.path.join(data_cache_path, data_id), 'r', encoding='utf-8') as f:
                texts = f.readlines()
            dataset = SubtitlesDataset(tokenizer=self.tokenizer, texts=texts)
        else:
            print(f"Data not in cache: {data_cache_path, data_id}")
            # Data not in cache, create it
            model_name = 'ai-forever/mGPT'
            remove_diacritics_path = '/cluster/work/cotterell/gacampa/MIT_prosody/data/mapping/remove_diacritics.pkl'
            #block_size = 128
            max_length = 256


            # import dataset
            dataset = load_dataset("open_subtitles", lang1="en", lang2="vi", split="train[:2000000]")
            with open(remove_diacritics_path, 'rb') as handle:
                vietnamese_mapping = pickle.load(handle)

            # remove diacritics from vietnamese sentences
            def map_vietnamese_letters(example):
                sentence = example['translation.vi']
                words = sentence.split()
                example['translation.vi'] = ' '.join(''.join(vietnamese_mapping.get(letter, letter) for letter in word) for word in words)
                return example
            
            dataset = dataset.flatten()
            dataset = dataset.map(map_vietnamese_letters, batched=False, 
                                            remove_columns=['id', 'meta.year', 'meta.imdbId', 'meta.subtitleId.en',
                                                            'meta.subtitleId.vi', 'meta.sentenceIds.en', 'meta.sentenceIds.vi', 
                                                            'translation.en'])
            
            print('First mapping completed')

            # import model and tokenizer
            tokenizer = mGPTTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(model_name)
            model.train()

            # create dataset
            def preprocess_function(examples):
                #return tokenizer([examples["translation.vi"]], padding=True, truncation=True, return_tensors="pt")
                return tokenizer([" ".join(x) for x in examples["translation.vi"]], padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
                
            dataset = dataset.map(preprocess_function, batched=True)
            dataset.set_format("pt", columns=["input_ids","attention_mask"], output_all_columns=True)
            print('Second mapping completed')

            '''def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # Drop the small remainder
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                # Split by chunks of block_size.
                result = {
                    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            dataset = dataset.map(group_texts, batched=True)
            print('Last mapping completed')'''
            train_dataset = dataset.train_test_split(test_size=0.1, seed=42)['train']
            if file_name == 'train-clean-100':
                dataset = train_dataset.train_test_split(test_size=0.1, seed=42)['train']
            elif file_name == 'test-clean':
                dataset = dataset.train_test_split(test_size=0.1, seed=42)['test']
            elif file_name == 'dev-clean':
                dataset = train_dataset.train_test_split(test_size=0.1, seed=42)['test']
            else:
                raise ValueError("File name not recognized.")
            
            texts = dataset["translation.vi"]
            dataset = dataset.remove_columns("translation.vi")

        #dataset = SubtitlesDataset(tokenizer=self.tokenizer, texts=texts)
        return texts, dataset

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
            print(self.hparams.model_name)
            if "gpt2" in self.hparams.model_name:
                print(f"Using gpt2 letter tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hparams.model_name, add_prefix_space=True
                )
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif "mGPT" in self.hparams.model_name:
                print(f"Using mGPT letter tokenizer")
                self.tokenizer = mGPTTokenizer.from_pretrained(self.hparams.model_name)
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError("Model name not recognized.")
        '''try:
            self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        except:'''
        self.pad_token_id = self.tokenizer.pad_token_id
        print(f"Dataloader: padding with token id: {self.pad_token_id}")

        self.train_texts, self.train_dataset = self.prepare_dataset(self.hparams.train_file)
        self.val_texts, self.val_dataset = self.prepare_dataset(self.hparams.val_file)
        self.test_texts, self.test_dataset = self.prepare_dataset(self.hparams.test_file)

        print('train_dataset',self.train_dataset)
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    # def encode_collate(self, batch):
    #     return encode_and_pad_batch(batch, self.tokenizer, self.hparams.model_name)

    def collate(self, batch):
        return vector_collate_fn(batch,self.pad_token_id)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            #collate_fn=self.collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            #collate_fn=self.collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            #collate_fn=self.collate,
            shuffle=False,
        )
