
# Implementation of the TrainingDataset class
from typing import Tuple
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
# import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# Classes defined by this application
from BilingualDataset import BilingualDataset

SOS_EOS_TOKEN_LEN = 2

class TrainingDataset:
    # Constructor
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer_dir = None
        self._tokenizer_src = None
        self._tokenizer_tgt = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._val_dataset = None

    def __get_sentences(self, ds : Dataset, lang : str) -> str:
        for item in ds:
            yield item['translation'][lang]

    def __create_lang_tokenizer(self, ds : Dataset, lang : str, max_len : str) -> Tokenizer:
        # This will create the tokenizer directory if it does not
        # exist but will not raise an error if the directory already
        # exists
        tokenizer_dir = Path(self._tokenizer_dir)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_fname = Path(f"{self._tokenizer_dir}/tokenizer_{lang}.json")
        if not Path.exists(tokenizer_fname):
            # No tokenizer file found. Create it from scratch.
            # See: 
            #   https://huggingface.co/docs/tokenizers/python/latest/api/reference.html
            #   https://huggingface.co/docs/transformers/en/tokenizer_summary
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.enable_truncation(max_length = max_len - SOS_EOS_TOKEN_LEN)
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(self.__get_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_fname))
        else:
            # The tokenizer JSON file exists. Load the tokenizer from the file that was previously created.
            tokenizer = Tokenizer.from_file(str(tokenizer_fname))

        return tokenizer

    def __get_max_seq_len(self, ds_raw, lang_src, lang_tgt) -> Tuple[int, int]:
        # Find the maximum length of each sentence in the source and target sentence
        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = self._tokenizer_src.encode(item['translation'][lang_src]).ids
            tgt_ids = self._tokenizer_tgt.encode(item['translation'][lang_tgt]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))            

        return max_len_src, max_len_tgt

    def get_language_dataset(self, config : dict) -> dict:
        # Get the model configuration parameters
        datasource = config["datasource"]
        lang_src = config["lang_src"]
        lang_tgt = config["lang_tgt"]
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]
        self._tokenizer_dir = config["tokenizer_dir"]

        # Load the dataset that contains the source language and its translation (or target) language
        # These datasets only have the training split
        #
        # See https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-fr for an example dataset
        #
        ds_raw = load_dataset(datasource, f"{lang_src}-{lang_tgt}", split='train')
        # ds_raw = load_dataset(datasource, f"{lang_src}-{lang_tgt}", split='train[0:100]')

        # Create the tokenizers for the source and target languages
        self._tokenizer_src = self.__create_lang_tokenizer(ds_raw, lang_src, seq_len)
        self._tokenizer_tgt = self.__create_lang_tokenizer(ds_raw, lang_tgt, seq_len)

        max_len_src, max_len_tgt = self.__get_max_seq_len(ds_raw, lang_src, lang_tgt)

        # Split the dataset: 90% for training and 10% for validation
        ds_raw_len = len(ds_raw)
        train_ds_size = int(0.9 * ds_raw_len)
        val_ds_size = ds_raw_len - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        # Save the validation dataset for probing the Transformer
        self._val_dataset = val_ds_raw

        # Create the Bilingual dataset
        train_ds = BilingualDataset(train_ds_raw, self._tokenizer_src, self._tokenizer_tgt, lang_src, lang_tgt, seq_len)
        val_ds = BilingualDataset(val_ds_raw, self._tokenizer_src, self._tokenizer_tgt, lang_src, lang_tgt, seq_len)

        self._train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self._val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

        dataset_dict = {
            "train_dataloader" : self._train_dataloader,
            "val_dataloader"   : self._val_dataloader,
            "val_dataset"      : self._val_dataset,
            "src_tokenizer"    : self._tokenizer_src,
            "tgt_tokenizer"    : self._tokenizer_tgt,
            "max_len_src"      : max_len_src,
            "max_len_tgt"      : max_len_tgt
        }

        return dataset_dict

    def src_vocab_size(self) -> int:
        return self._tokenizer_src.get_vocab_size()

    def tgt_vocab_size(self) -> int:
        return self._tokenizer_tgt.get_vocab_size()

