#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:08
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   dataloader4torch.py
# @Desc     :
from click.core import batch
from torch import Tensor, tensor, float32
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class TorchDataLoader:
    """ A custom PyTorch DataLoader class for handling TorchDataset """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32, shuffle_state: bool = True,
                 workers: int = 0,
                 batch_pad: bool = False, PAD: int = 0, batch_first: bool = True, padding_direction: str = "right"
                 ):
        """ Initialise the TorchDataLoader class
        :param dataset: the TorchDataset or Dataset to load data from
        :param batch_size: the number of samples per batch
        :param shuffle_state: whether to shuffle the data at every epoch
        :param workers: the number of workers to use for data loading
        :param batch_pad: whether to pad sequences in the batch
        :param PAD: the padding value for sequences
        :param batch_first: whether to have batch dimension first
        :param padding_direction: side to apply padding ("right" or "left")
        """
        self._dataset: Dataset = dataset
        self._batches: int = batch_size
        self._shuffle: bool = shuffle_state
        self._batch_pad: bool = batch_pad
        self._PAD = PAD
        self._first: bool = batch_first
        self._direction: str = padding_direction

        pin_memory: bool = False if workers == 0 else True
        print(f"num_workers={workers} | pin_memory={pin_memory}")
        print()

        batch_func = self._collate_fn if self._batch_pad else None
        self._loader: DataLoader = DataLoader(
            dataset=self._dataset,
            batch_size=self._batches,
            shuffle=self._shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=batch_func
        )

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def _collate_fn(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """ Custom collate function to process a batch of data """
        # batch: list[tuple[Tensor, Tensor]]
        features: list[Tensor] = [tensor(f, dtype=float32) for f, _ in batch]
        labels: list[Tensor] = [tensor(l, dtype=float32) for _, l in batch]

        features: Tensor = pad_sequence(
            features,
            batch_first=self._first, padding_value=self._PAD, padding_side=self._direction
        )
        labels: Tensor = pad_sequence(
            labels,
            batch_first=self._first, padding_value=self._PAD, padding_side=self._direction
        )

        return features, labels

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """ Return a single (feature, label) pair or a batch via slice """
        if not isinstance(index, int):
            raise TypeError(f"Invalid index type: {type(index)}")

        if self._batch_pad:
            batch_start = (index // self._batches) * self._batches
            batch_end = min(batch_start + self._batches, len(self._dataset))
            batch = [self._dataset[i] for i in range(batch_start, batch_end)]

            features, labels = self._collate_fn(batch)

            batch_idx = index - batch_start
            return features[batch_idx], labels[batch_idx]
        else:
            return self._dataset[index]

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def __repr__(self):
        return (f"TorchDataLoader(dataset={self._dataset}, "
                f"batch_size={self._batches}, "
                f"shuffle={self._shuffle}), "
                f"pad_value={self._PAD}, "
                f"batch_first={self._first}, "
                f"padding_side='{self._direction}')")


if __name__ == "__main__":
    pass
