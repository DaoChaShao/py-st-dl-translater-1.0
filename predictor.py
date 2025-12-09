#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from pathlib import Path
from random import randint
from re import search
from torch import Tensor, load, device, no_grad, nn, tensor, long, argmax

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import Lang, Tokens, Tasks
from src.nets.gru import GRUNet
from src.utils.apis import OpenAITextCompleter
from src.utils.helper import Timer, read_yaml
from src.utils.highlighter import red, green, yellow, blue, purple
from src.utils.nlp import spacy_batch_tokeniser
from src.utils.PT import item2tensor
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
        amount: int | None = 100
        # amount: int | None = None
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

        # Convert a random sentence to sequence using dictionary
        idx: int = randint(0, len(news) - 1)
        sequence: list[str] = news[idx]
        seq: list[int] = [dictionary.get(item, Tokens.UNK) for item in sequence]
        # print(seq)

        # Convert the token to a tensor
        X: Tensor = item2tensor(seq, embedding=True, accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR)
        # Add batch size
        X = X.unsqueeze(0)
        # print(X)
        # Get the relevant label
        labels: list[int] = [label for label, _ in data]
        y: Tensor = item2tensor(labels[idx], accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR)
        print(y)

        # Load the save model parameters
        params: Path = Path(CONFIG4RNN.FILEPATHS.SAVED_NET)
        if params.exists():
            print(f"Model {params.name} Exists!")

            # Set up a model and load saved parameters
            model = GRUNet(
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

            # Prediction
            with no_grad():
                logits: Tensor = model(X)
                # print(logist)
                probabilities: Tensor = nn.functional.softmax(logits)
                # print(probabilities)
                y_pred: Tensor = argmax(probabilities, dim=1)
                print(y_pred)

                correct: bool = (y_pred.item() == y.item())
                # print(correct)
                print(blue("Positive") if y == 2 else purple("Negative") if y == 1 else yellow("Neutral"))

                rate: str = green("Bingo") if correct == True else red("Damn")
                print(f"The content of the selected news {idx}:\n{contents[idx]}")
                print(f"The prediction of the news {idx} is {rate} !")

            # Prompt Engineering with OpenAI API
            key: Path = Path(CONFIG4RNN.FILEPATHS.API_KEY)
            config: dict = read_yaml(key)
            API_KEY: str = config["openai"]["api_key"]
            opener = OpenAITextCompleter(API_KEY, temperature=0)
            role: str = "You are a professional financial expert with cross-cultural expertise."
            rating: str = """
                        0: Neutral Review
                        1: Negative Review
                        2: Positive Review
                        """
            prompt: str = f"""
                        Give a brief explanation in Chinese after reading the following review:
                        {contents[idx]}.
                        Please analyze it, give a reason, and provide a rating (Only return number) as follows:
                        {rating}.
                        """
            explanation = opener.client(role, prompt)
            print(explanation)

            match = search(r"\b[0-2]\b", explanation)
            pred_label: int | None = int(match.group()) if match else None
            print(f"Predicted Label from OpenAI: {pred_label}")

            result = green("Bingo") if (pred_label == y_pred) else red("Damn")
            print(f"The prediction between OpenAI and RNN Model: {result} !")
        else:
            print(f"Model {params.name} does not exist!")


if __name__ == "__main__":
    main()
