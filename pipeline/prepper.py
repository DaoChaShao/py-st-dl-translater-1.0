#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   prepper.py
# @Desc     :   

from numpy import ndarray
from pathlib import Path
from random import randint

from src.configs import Tokens
from src.configs.cfg_dl import CONFIG4DL
from src.datasets.seq_classification import TorchDataset4Seq2Classification
from src.dataloaders.dataloader4torch import TorchDataLoader
from src.utils.stats import load_json

from pipeline.processor import process_data


def prepare_data() -> tuple[TorchDataLoader, TorchDataLoader, int, ndarray]:
    """ Prepare data """
    # Get data
    features4train, labels4train, features4valid, labels4valid, MAX_SEQ_LEN, computed_weights = process_data()

    # Load dictionary
    dic: Path = Path(CONFIG4DL.FILEPATHS.DICTIONARY)
    dictionary: dict = load_json(dic) if dic.exists() else print("Dictionary file not found.")

    # Set dataset
    dataset4train = TorchDataset4Seq2Classification(
        features4train, labels4train,
        seq_max_len=MAX_SEQ_LEN,
        pad_token=dictionary[Tokens.PAD]
    )
    dataset4valid = TorchDataset4Seq2Classification(
        features4valid, labels4valid,
        seq_max_len=MAX_SEQ_LEN,
        pad_token=dictionary[Tokens.PAD]
    )
    # idx4train: int = randint(0, len(dataset4train) - 1)
    # print(dataset4train[idx4train])
    # idx4valid: int = randint(0, len(dataset4valid) - 1)
    # print(dataset4valid[idx4valid])
    # print()

    # Set up dataloader
    dataloader4train = TorchDataLoader(
        dataset4train,
        batch_size=CONFIG4DL.PREPROCESSOR.BATCHES,
        shuffle_state=CONFIG4DL.PREPROCESSOR.SHUFFLE,
        workers=CONFIG4DL.PREPROCESSOR.WORKERS,
    )
    dataloader4valid = TorchDataLoader(
        dataset4valid,
        batch_size=CONFIG4DL.PREPROCESSOR.BATCHES,
        shuffle_state=CONFIG4DL.PREPROCESSOR.SHUFFLE,
        workers=CONFIG4DL.PREPROCESSOR.WORKERS,
    )
    # idx4train: int = randint(0, len(dataloader4train) - 1)
    # print(dataloader4train[idx4train])
    # idx4valid: int = randint(0, len(dataloader4valid) - 1)
    # print(dataloader4valid[idx4valid])
    # print()

    return dataloader4train, dataloader4valid, MAX_SEQ_LEN, computed_weights


if __name__ == "__main__":
    prepare_data()
