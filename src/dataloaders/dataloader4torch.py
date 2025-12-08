#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:08
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   dataloader4torch.py
# @Desc     :

from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class TorchDataLoader:
    """ A custom PyTorch DataLoader class for handling TorchDataset """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32, shuffle_state: bool = True,
                 workers: int = 0,
                 ):
        """ Initialise the TorchDataLoader class
        :param dataset: the TorchDataset or Dataset to load data from
        :param batch_size: the number of samples per batch
        :param shuffle_state: whether to shuffle the data at every epoch
        :param workers: the number of workers to use for data loading
        """
        self._dataset: Dataset = dataset
        self._batches: int = batch_size
        self._shuffle: bool = shuffle_state

        pin_memory: bool = False if workers == 0 else True
        print(f"num_workers={workers} | pin_memory={pin_memory}")
        print()
        self._loader: DataLoader = DataLoader(
            dataset=self._dataset,
            batch_size=self._batches,
            shuffle=self._shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
        )

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """ Return a single (feature, label) pair or a batch via slice """
        if not isinstance(index, int):
            raise TypeError(f"Invalid index type: {type(index)}")
        return self._dataset[index]

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def __repr__(self):
        return (f"TorchDataLoader(dataset={self._dataset}, "
                f"batch_size={self._batches}, "
                f"shuffle={self._shuffle})")


if __name__ == "__main__":
    pass
