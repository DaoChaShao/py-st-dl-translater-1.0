#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 19:28
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   processor.py
# @Desc     :   

from numpy import ndarray, array
from pathlib import Path
from pprint import pprint
from random import randint

from tqdm import tqdm

from src.configs.cfg_dl import CONFIG4DL
from src.configs.cfg_types import Language, Tokens
from src.utils.helper import Timer
from src.utils.highlighter import starts, lines
from src.utils.nlp import spacy_batch_tokeniser, count_frequency, build_word2id_seqs, check_vocab_coverage
from src.utils.PT import balance_imbalanced_weights
from src.utils.stats import create_full_data_split, save_json
from src.utils.SQL import SQLiteIII

from pipeline import preprocess_data


def process_data() -> tuple[list, list, list, list, int, ndarray]:
    """ Main Function """
    with Timer("Process Data"):
        # Get the data from the database
        table: str = "news"
        cols: dict[str, type[int | str]] = {"label": int, "content": str}
        sqlite = SQLiteIII(table, cols, CONFIG4DL.FILEPATHS.SQLITE)
        sqlite.connect()
        data: list[tuple] = sqlite.fetch_all([col for col in cols.keys()])
        sqlite.close()
        # pprint(data[:3])
        # print(len(data))

        # Get imbalanced category result
        labels: list[int] = [label for label, _ in data]
        unique_labels: list[int] = list(set(labels))
        print(unique_labels)
        computed_weights: ndarray = balance_imbalanced_weights(labels, list(set(labels)))
        # print(computed_weights)

        # Separate the data
        data4train, data4valid, _ = create_full_data_split(data)
        # print(data4train[:3])
        # print(data4valid[:3])

        # Set a dictionary
        # amount: int | None = 100
        amount: int | None = None
        contents4train: list[str] = [content for _, content in data4train]
        contents4valid: list[str] = [content for _, content in data4valid]
        if amount is None:
            items4train: list[list[str]] = spacy_batch_tokeniser(
                contents4train, lang=Language.EN, batches=CONFIG4DL.PREPROCESSOR.BATCHES
            )
            items4valid: list[list[str]] = spacy_batch_tokeniser(
                contents4valid, lang=Language.EN, batches=CONFIG4DL.PREPROCESSOR.BATCHES
            )
        else:
            items4train: list[list[str]] = spacy_batch_tokeniser(
                contents4train[:amount], lang=Language.EN, batches=CONFIG4DL.PREPROCESSOR.BATCHES
            )
            items4valid: list[list[str]] = spacy_batch_tokeniser(
                contents4valid[:amount], lang=Language.EN, batches=CONFIG4DL.PREPROCESSOR.BATCHES
            )
        # print(items4valid[:3])
        # print(items4train[:3])

        # Count the frequency of the words
        words4train: list[str] = [word for sentence in items4train for word in sentence]
        words4valid: list[str] = [word for sentence in items4valid for word in sentence]
        words4all: list[str] = words4train + words4valid
        # print(words[:3])
        tokens, _ = count_frequency(words4all, top_k=10, freq_threshold=3)
        # print(tokens[:10])

        # Build a dictionary
        special: list[str] = [Tokens.PAD, Tokens.UNK, Tokens.SOS, Tokens.EOS]
        # print(special)
        dictionary: dict[str, int] = {
            word: i for i, word in
            tqdm(enumerate(special + tokens), total=len(special + tokens), desc="Building dictionary")
        }
        save_json(dictionary, CONFIG4DL.FILEPATHS.DICTIONARY)
        dic: Path = Path(CONFIG4DL.FILEPATHS.DICTIONARY)
        print("Dictionary Saved Successfully!") if dic.exists() else print("Dictionary NOT Saved!")

        # Build sequence for train
        seq4train: list[list[int]] = build_word2id_seqs(items4train, dictionary, Tokens.UNK)
        # idx4train: int = randint(0, len(seq4train) - 1)
        # print(seq4train[idx4train])
        # Build sequence for all, train and valid
        seq4valid: list[list[int]] = build_word2id_seqs(items4valid, dictionary, Tokens.UNK)
        # idx4valid: int = randint(0, len(seq4valid) - 1)
        # print(seq4valid[idx4valid])
        # Build sequence for all, train and valid
        seq4all: list[list[int]] = seq4train + seq4valid
        # idx4all: int = randint(0, len(seq4all) - 1)
        # print(seq4all[idx4all])

        # Get the train dataset sentences description
        lengths: list[int] = [len(seq) for seq in seq4all]
        max_len: int = max(lengths)
        min_len: int = min(lengths)
        avg_len: float = sum(lengths) / len(lengths)
        # Check the coverage of train data
        check_vocab_coverage(words4train, dictionary)
        # Check the coverage of valid data
        check_vocab_coverage(words4valid, dictionary)

        # Zip labels and contents back
        if amount is None:
            labels4train: list[int] = [label for label, _ in data4train]
            labels4valid: list[int] = [label for label, _ in data4valid]
        else:
            labels4train: list[int] = [label for label, _ in data4train[:amount]]
            labels4valid: list[int] = [label for label, _ in data4valid[:amount]]
        assert len(seq4train) == len(labels4train), "Train Labels and Contents Length Mismatch!"
        assert len(seq4valid) == len(labels4valid), "Valid Labels and Contents Length Mismatch!"
        # print(label4train[:3])
        # print(label4valid[:3])

        starts()
        print("Data Preprocessing Summary:")
        lines()
        print(f"Train dataset: {len(data4train)} Samples")
        print(f"Valid dataset: {len(data4valid)} Samples")
        print(f"Dictionary Size: {len(dictionary)}")
        print(f"The min length of the sequence: {min_len}")
        print(f"The average length of the sequence: {avg_len:.2f}")
        print(f"The max length of the sequence: {max_len}")
        starts()
        print()
        """
        ****************************************************************
        Data Preprocessing Summary:
        ----------------------------------------------------------------
        Train dataset: 44089 Samples
        Valid dataset: 9448 Samples
        Dictionary Size: 9570
        The min length of the sequence: 0
        The average length of the sequence: 12.83
        The max length of the sequence: 48
        ****************************************************************
        """

        return seq4train, labels4train, seq4valid, labels4valid, max_len, computed_weights


if __name__ == "__main__":
    process_data()
