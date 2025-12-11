#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   evaluator.py
# @Desc     :

from pathlib import Path
from random import randint
from torch import Tensor, load, device, no_grad, nn, argmax
from tqdm import tqdm

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import Lang, Tokens, Tasks
from src.dataloaders.dataloader4torch import TorchDataLoader
from src.datasets.TS4classification import TimeSeriesTorchDatasetForClassification
from src.nets.gru4classification import GRUClassifier
from src.utils.helper import Timer
from src.utils.highlighter import starts, lines, red, green
from src.utils.nlp import spacy_batch_tokeniser, build_word2id_seqs
from src.utils.SQL import SQLiteIII
from src.utils.stats import create_full_data_split, load_json


def main() -> None:
    """ Main Function """
    # Get the data from the database: Method II
    table: str = "news"
    cols: dict[str, type[int | str]] = {"label": int, "content": str}
    with SQLiteIII(table, cols, CONFIG4RNN.FILEPATHS.SQLITE) as db:
        database = db.fetch_all([col for col in cols.keys()])
        # print(len(database))
        # print()
    # pprint(data[:3])
    # print(len(data))

    with Timer("Next Word Prediction"):
        # Separate the data
        _, _, data = create_full_data_split(database)

        # Tokenise the data
        # amount: int | None = 100
        amount: int | None = None
        contents: list[str] = [content for _, content in data]
        if amount is None:
            news: list[list[str]] = spacy_batch_tokeniser(
                contents, lang=Lang.EN, batches=CONFIG4RNN.PREPROCESSOR.BATCHES
            )
        else:
            news: list[list[str]] = spacy_batch_tokeniser(
                contents[:amount], lang=Lang.EN, batches=CONFIG4RNN.PREPROCESSOR.BATCHES
            )
        # print(news)

        # Load the dictionary and convert
        dic: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY)
        dictionary: dict[str, int] = load_json(dic)

        # Convert sentences to sequence using dictionary

        sequences: list[list[int]] = build_word2id_seqs(news, dictionary, Tokens.UNK)

        # Get the relevant label
        labels: list[int] = [label for label, _ in data]

        # Set up dataset
        MAX_SEQ_LEN: int = 48
        dataset = TimeSeriesTorchDatasetForClassification(
            sequences, labels,
            seq_max_len=MAX_SEQ_LEN,
            pad_token=dictionary[Tokens.PAD]
        )
        # starts()
        # print("Data Preprocessing Summary:")
        # lines()
        # print(f"Test dataset: {len(dataset)} Samples")
        # print(f"Dictionary Size: {len(dictionary)}")
        # starts()
        # Set up dataloader
        dataloader = TorchDataLoader(
            dataset,
            batch_size=CONFIG4RNN.PREPROCESSOR.BATCHES,
            shuffle_state=CONFIG4RNN.PREPROCESSOR.SHUFFLE,
            workers=CONFIG4RNN.PREPROCESSOR.WORKERS,
        )
        idx: int = randint(0, len(dataloader) - 1)
        # print(dataloader[idx])

        # Load the save model parameters
        params: Path = Path(CONFIG4RNN.FILEPATHS.SAVED_NET)
        if params.exists():
            print(f"Model {params.name} Exists!")

            # Set up a model and load saved parameters
            model = GRUClassifier(
                len(dictionary),
                embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
                hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
                num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
                num_classes=CONFIG4RNN.PARAMETERS.CLASSES,
                dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
                accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
                task=Tasks.CLASSIFICATION,
                pad_idx=dictionary[Tokens.PAD]
            )
            dict_state: dict = load(params, map_location=device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
            model.load_state_dict(dict_state)
            model.eval()
            print("Model Loaded Successfully!")

            # Predict
            total = 0
            correct_counter = 0

            with no_grad():
                for X, y in tqdm(dataloader, total=len(dataloader), desc="Evaluating Trained Model:"):
                    logits = model(X)
                    # Convert the output to probabilities along the class dimension
                    probabilities: Tensor = nn.functional.softmax(logits, dim=1)
                    # Get predicted labels
                    y_pred: Tensor = argmax(probabilities, dim=1)

                    # Compare with true labels
                    correct: Tensor = (y_pred == y)

                    # Update counters
                    total += y.size(0)
                    correct_counter += correct.sum().item()

            # Accuracy
            accuracy = correct_counter / total
            starts()
            print("Trained Model Evaluation")
            lines()
            print(f"Total Samples: {total}")
            print(f"Correct Predictions: {correct_counter}")
            print(f"Accuracy: {accuracy:.2%}")
            starts()
            """
            ****************************************************************
            Trained LSTM Model Evaluation
            ----------------------------------------------------------------
            Total Samples: 9448
            Correct Predictions: 9440
            Accuracy: 99.92%
            ****************************************************************
            ****************************************************************
            Trained GRU Model Evaluation
            ----------------------------------------------------------------
            Total Samples: 9448
            Correct Predictions: 9440
            Accuracy: 99.92%
            ****************************************************************
            """
        else:
            print(f"Model {params.name} does not exist!")


if __name__ == "__main__":
    main()
