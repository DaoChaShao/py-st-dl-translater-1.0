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
from src.configs.cfg_types import Tokens, Seq2SeqNet
from src.configs.parser import set_argument_parser
from src.trainers.trainer4torch import TorchTrainer
from src.nets.seq2seq import SeqToSeqCoder
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
        dic_cn: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY_CN)
        dictionary_cn = load_json(dic_cn) if dic_cn.exists() else print("Dictionary file not found.")
        dic_en: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY_EN)
        dictionary_en = load_json(dic_en) if dic_en.exists() else print("Dictionary file not found.")
        # print(dictionary_cn[Tokens.PAD])
        # print(dictionary_en[Tokens.PAD])

        # Get the input size and number of classes
        vocab_size4cn: int = len(dictionary_cn)
        vocab_size4en: int = len(dictionary_en)
        print(vocab_size4cn, vocab_size4en)

        # Get the data
        train, valid = prepare_data()

        # Initialize model
        model = SeqToSeqCoder(
            vocab_size4input=vocab_size4cn,
            vocab_size4output=vocab_size4en,
            embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
            hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
            num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
            dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
            bid=True,
            pad_idx4input=dictionary_cn[Tokens.PAD],
            pad_idx4output=dictionary_en[Tokens.PAD],
            net_category=Seq2SeqNet.GRU,
        )
        # Setup optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=args.alpha, weight_decay=CONFIG4RNN.HYPERPARAMETERS.DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(ignore_index=CONFIG4RNN.PARAMETERS.PAD_LABELS_IN_BATCH)
        model.summary()
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
            log_name=Seq2SeqNet.GRU
        )


if __name__ == "__main__":
    main()
