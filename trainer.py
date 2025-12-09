#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/2 22:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   trainer.py
# @Desc     :   

from pathlib import Path
from torch import optim, nn, Tensor

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import Tasks, Tokens
from src.configs.parser import set_argument_parser
from src.trainers.trainer4torch import TorchTrainer
from src.nets.gru import GRUNet
from src.utils.PT import item2tensor
from src.utils.stats import load_json
from src.utils.PT import TorchRandomSeed

from pipeline.prepper import prepare_data


def main() -> None:
    """ Main Function """
    # Set up argument parser
    args = set_argument_parser()

    with TorchRandomSeed("Financial News Classification"):
        # Get the dictionary
        dic: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY)
        dictionary = load_json(dic) if dic.exists() else print("Dictionary file not found.")
        # print(dictionary[Tokens.PAD])

        # Get the data
        train, valid, MAX_SEQ_LEN, balanced_weights = prepare_data()

        # Get the input size and number of classes
        vocab_size: int = len(dictionary)

        # Initialize model
        model = GRUNet(
            vocab_size=vocab_size,
            embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
            hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
            num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
            num_classes=CONFIG4RNN.PARAMETERS.CLASSES,
            dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
            accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
            task=Tasks.CLASSIFICATION,
            pad_idx=dictionary[Tokens.PAD]
        )
        """
        ================================================================
        Model Summary for LSTMRNNForClassification
        ----------------------------------------------------------------
        - Vocabulary size: 9570
        - Embedding dim: 128
        - Hidden size: 256
        - Num layers: 2
        - Output classes: 3
        - Total parameters: 3,593,987
        - Trainable parameters: 3,593,987
        ================================================================
        """

        # Set up balanced weights among different classes if needed
        computed_weights: Tensor = item2tensor(balanced_weights, accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR)

        # Setup optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=args.alpha, weight_decay=CONFIG4RNN.HYPERPARAMETERS.DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(weight=computed_weights)
        model.summary()

        # Setup trainer
        trainer = TorchTrainer(
            model=model,
            optimiser=optimizer,
            criterion=criterion,
            accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
            scheduler=scheduler,
        )
        # Train the model
        trainer.fit(
            train_loader=train,
            valid_loader=valid,
            epochs=args.epochs,
            model_save_path=str(CONFIG4RNN.FILEPATHS.SAVED_NET),
            log_name="GRU-weights"
        )


if __name__ == "__main__":
    main()
