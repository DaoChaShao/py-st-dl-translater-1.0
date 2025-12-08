#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preprocessor.py
# @Desc     :   

from numpy import array, ndarray
from pandas import DataFrame, read_csv
from pprint import pprint
from pathlib import Path
from random import randint
from sklearn.utils.class_weight import compute_class_weight

from src.configs.cfg_base import CONFIG
from src.utils.helper import Timer
from src.utils.SQL import SQLiteIII


def preprocess_data() -> ndarray:
    """ Main Function """
    with Timer("Preprocess Data"):
        path: Path = Path(CONFIG.FILEPATHS.DATA4ALL)

        if path.exists():
            print(f"Bingo! {path.name} exists!")
            print()

            # Get raw structural data
            raw: DataFrame = read_csv(path, encoding="latin1")  # Or, use "ISO-8859-1"
            # print(raw.head())
            # print()
            headers: list[str] = ["label", "news"]
            raw.columns = headers
            # print(raw.head())
            # print()
            idx: int = randint(0, raw.shape[0] - 1)
            pprint(raw.iloc[idx])
            print(type(raw.iloc[idx]), raw.shape)
            print()

            # Check the categories
            print(raw["label"].value_counts())
            print()

            # Convert string labels to integer labels
            categories: dict = {label: idx for idx, label in enumerate(raw["label"].unique())}
            # print(categories)
            # print()
            """
            categories = {'neutral': 0, 'negative': 1, 'positive': 2}
            """
            raw["label"] = raw["label"].map(categories)
            pprint(raw.iloc[idx])
            print(type(raw.iloc[idx]), raw.shape)
            print()

            # Check and balance the weights of different classes
            balanced_weights = compute_class_weight(
                class_weight="balanced",
                classes=array(list(categories.values())),
                y=raw["label"].to_numpy(),
            )
            print(balanced_weights)
            print()

            # Get the news
            news: list[str] = raw["news"].tolist()
            labels: list[int] = raw["label"].tolist()
            print(news[idx])
            print(labels[idx])

            # Store the preprocessed data into sqlit 3 database
            table: str = "news"
            cols: dict[str, type[int | str]] = {"label": int, "content": str}
            data: dict[str, list[int | str]] = {"label": labels, "content": news}
            with SQLiteIII(table, cols, CONFIG.FILEPATHS.SQLITE) as db:
                db.insert(data)
        else:
            print(f"{path.name} does NOT exist!")

        return balanced_weights


if __name__ == "__main__":
    preprocess_data()
