#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 23:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   dataset4torch.py
# @Desc     :

from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor, tensor, float32
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    """ A custom PyTorch Dataset class for handling label features and labels """

    def __init__(self, features, labels, batch_pad: bool = False):
        """ Initialise the TorchDataset class
        :param features: raw features(list/ndarray/DataFrame) or padded Tensor
        :param labels: raw labels(list/ndarray/DataFrame) or padded Tensor
        :param batch_pad: if True → keep raw lists, collate_fn will pad them
        """
        self._batch_pad = batch_pad

        if self._batch_pad:
            # Do not convert to tensor here; will be handled in collate_fn
            self._features = features
            self._labels = labels
        else:
            self._features: Tensor = self._to_tensor(features)
            self._labels: Tensor = self._to_tensor(labels)

    @property
    def features(self) -> Tensor | list:
        """ Return the feature tensor as a property """
        return self._features

    @property
    def labels(self) -> Tensor | list:
        """ Return the label tensor as a property """
        return self._labels

    @staticmethod
    def _to_tensor(data: DataFrame | Tensor | ndarray | list) -> Tensor:
        """ Convert input data to a PyTorch tensor on the specified device
        :param data: the input data (DataFrame, ndarray, list, or Tensor)
        :return: the converted PyTorch tensor
        """
        if isinstance(data, (DataFrame, Series)):
            out = tensor(data.values, dtype=float32)
        elif isinstance(data, Tensor):
            out = data.float()
        elif isinstance(data, (ndarray, list)):
            out = tensor(data, dtype=float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return out

    def __len__(self) -> int:
        """ Return the total number of samples in the dataset """
        return len(self._features)

    def __getitem__(self, index: int | slice) -> tuple[Tensor, Tensor] | tuple[list, list]:
        """ Return a single or sliced (feature, label) pair """
        return self._features[index], self._labels[index]

    def __repr__(self):
        """ Return a string representation of the dataset """
        if self._batch_pad:
            info4features = f"len={len(self._features)} (unpadded list)"
            info4labels = f"len={len(self._labels)} (unpadded list)"
        else:
            info4features = f"shape={tuple(self._features.shape)}"
            info4labels = f"shape={tuple(self._labels.shape)}"

        return f"TorchDataset(features={info4features}, labels={info4labels})"


if __name__ == "__main__":
    pass
