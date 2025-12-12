#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/11 16:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seq2seq.py
# @Desc     :   

from torch import (nn, Tensor,
                   full, long, cat,
                   randint, )

from src.utils.highlighter import starts, lines

WIDTH: int = 64


class SeqEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bid: bool = True,
                 pad_idx: int = 0,
                 net_category: str = "gru",
                 ):
        super().__init__()
        """ Initialise the Encoder class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bid: bidirectional flag
        :param pad_idx: padding index for the embedding layer
        :param net_category: network category (e.g., 'gru')
        """
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

        self._L = vocab_size  # Lexicon/Vocabulary size for encoder / input
        self._H = embedding_dim  # Embedding dimension
        self._M = hidden_size  # Hidden dimension
        self._C = num_layers  # RNN layers count
        self._type = net_category  # Network category

        self._embedder = nn.Embedding(self._L, self._H, padding_idx=pad_idx)
        self._net = nets[net_category](
            self._H, self._M, num_layers,
            batch_first=True, bidirectional=bid,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )

    def forward(self, src: Tensor) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        embedded = self._embedder(src)
        lengths: Tensor = (src != self._embedder.padding_idx).sum(dim=1)

        result = self._net(embedded)

        if self._type == "lstm":
            outputs, (hidden, cell) = result
            return outputs, (hidden, cell), lengths
        else:
            outputs, hidden = result
            return outputs, hidden, lengths


class SeqDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, pad_idx: int = 0,
                 net_category: str = "gru",
                 ):
        super().__init__()
        """ Initialise the Decoder class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param pad_idx: padding index for the embedding layer
        :param net_category: network category (e.g., 'gru')
        """
        nets: dict[str, type] = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

        self._L = vocab_size  # Lexicon/Vocabulary size
        self._H = embedding_dim  # Embedding dimension
        self._M = hidden_size  # Hidden dimension
        self._C = num_layers  # RNN layers count
        self._type = net_category  # Network category

        self._embedder = nn.Embedding(self._L, self._H, padding_idx=pad_idx)
        self._net = nets[net_category](
            self._H, self._M, num_layers, batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self._dropout = nn.Dropout(p=dropout_rate if num_layers > 1 else 0.0)
        self._linear = nn.Linear(self._M, self._L)

    def forward(self, tgt: Tensor, hidden: Tensor | tuple[Tensor, Tensor]) -> tuple:
        embedded = self._embedder(tgt)

        if self._type == "lstm":
            outputs, (hn, cn) = self._net(embedded, hidden)
            logits = self._linear(self._dropout(outputs))
            return logits, (hn, cn)
        else:
            h = hidden[0] if isinstance(hidden, tuple) else hidden
            outputs, hn = self._net(embedded, h)
            logits = self._linear(self._dropout(outputs))
            return logits, (hn,)


class SeqToSeqCoder(nn.Module):
    """ An RNN model for sequence-to-sequence tasks using PyTorch """

    def __init__(self,
                 vocab_size4input: int, vocab_size4output: int,
                 embedding_dim: int, hidden_size: int, num_layers: int,
                 dropout_rate: float = 0.3, bid: bool = True,
                 pad_idx4input: int = 0, pad_idx4output: int = 0,
                 net_category: str = "gru"
                 ):
        super().__init__()
        """ Initialise the SeqToSeqRNN class
        :param vocab_size4input: size of the input vocabulary
        :param vocab_size4output: size of the output vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param dropout_rate: dropout rate for regularization
        :param bid: bidirectional flag for the encoder
        :param pad_idx4input: padding index for the input embedding layer
        :param pad_idx4output: padding index for the output embedding layer
        :param accelerator: computation accelerator (e.g., 'cpu', 'cuda')
        :param net_category: network category (e.g., 'gru')
        """
        self._L4IN = vocab_size4input  # Lexicon/Vocabulary size for encoder / input
        self._L4OUT = vocab_size4output  # Lexicon/Vocabulary size for decoder / output
        self._H = embedding_dim  # Embedding dimension
        self._M = hidden_size  # Hidden dimension
        self._C = num_layers  # RNN layers count
        self._bid = bid  # Bidirectional flag for encoder
        self._type = net_category  # Network category

        self._encoder = SeqEncoder(
            self._L4IN, self._H, self._M, self._C,
            dropout_rate=dropout_rate if num_layers > 1 else 0.0,
            bid=self._bid, pad_idx=pad_idx4input, net_category=net_category,
        )
        self._decoder = SeqDecoder(
            self._L4OUT, self._H, self._M, self._C,
            dropout_rate=dropout_rate if num_layers > 1 else 0.0,
            pad_idx=pad_idx4output, net_category=net_category,
        )

        self._init_weights()

    def _init_weights(self):
        """ Initialize model weights """
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """ Forward pass through the Seq2Seq model
        :param src: source/input tensor
        :param tgt: target/output tensor
        :return: output logits tensor
        """
        c = None

        # Encoder
        if self._type == "lstm":
            outputs, (h, c), lengths = self._encoder(src)
        else:
            outputs, h, lengths = self._encoder(src)

        # Decoder
        if self._bid:
            B = h.size(1)
            if self._type == "lstm":
                h = h.view(self._C, 2, B, self._M)
                c = c.view(self._C, 2, B, self._M)
                decoder_state = ((h[:, 0] + h[:, 1]) / 2, (c[:, 0] + c[:, 1]) / 2)
            else:
                h = h.view(self._C, 2, B, self._M)
                decoder_state = ((h[:, 0] + h[:, 1]) / 2,)
        else:
            decoder_state = (h, c) if self._type == "lstm" else (h,)

        decoder_input = tgt[:, :-1]
        logits, _ = self._decoder(decoder_input, decoder_state)

        return logits

    def summary(self):
        """ Print a summary of the model architecture and parameters """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("*" * WIDTH)
        print(f"Model: {self.__class__.__name__}")
        print("-" * WIDTH)
        print(f"Encoder Vocab Size: {self._L4IN}")
        print(f"Decoder Vocab Size: {self._L4OUT}")
        print(f"Embedding Dim: {self._H}")
        print(f"Hidden Size: {self._M}")
        print(f"Num Layers: {self._C}")
        print(f"Bidirectional Encoder: {self._bid}")
        print(f"RNN Type: {self._type}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("*" * WIDTH)

    def generate(self, src: Tensor, max_len: int = 50):
        """ Regression generate automatically """
        batch_size = src.size(0)

        # Encoder
        encoder_outputs, encoder_hidden, lengths = self._encoder(src)

        # Decoder
        if self._type == "lstm":
            h, c = encoder_hidden
            if self._bid:
                B = h.size(1)
                h = h.view(self._C, 2, B, self._M)
                c = c.view(self._C, 2, B, self._M)
                decoder_hidden = ((h[:, 0] + h[:, 1]) / 2, (c[:, 0] + c[:, 1]) / 2)
            else:
                decoder_hidden = (h, c)
        else:
            # GRU
            h = encoder_hidden
            if self._bid:
                B = h.size(1)
                h = h.view(self._C, 2, B, self._M)
                decoder_hidden = ((h[:, 0] + h[:, 1]) / 2,)  # tuple
            else:
                decoder_hidden = (h,)  # tuple

        # Start from SOS token (assumed to be index 1)
        decoder_input = full((batch_size, 1), 1, dtype=long, device=src.device)
        generated = []

        for _ in range(max_len):
            logits, decoder_hidden = self._decoder(decoder_input, decoder_hidden)
            # Greedy
            next_token = logits.argmax(dim=2)
            generated.append(next_token)

            # Assume all the next tokens is EOS, stop it
            if (next_token == 2).all():
                break

            decoder_input = next_token

        return cat(generated, dim=1)


if __name__ == "__main__":
    test_cases = [
        ("gru", True, "GRU-bid"), ("gru", False, "GRU-one"),
        ("lstm", True, "LSTM-bid"), ("lstm", False, "LSTM-one"),
        ("rnn", True, "RNN-bid"), ("rnn", False, "RNN-one"),
    ]

    for rnn_type, bid, desc in test_cases:
        starts()
        print(f"Test: {desc}")
        lines()

        try:
            model = SeqToSeqCoder(
                vocab_size4input=5000,
                vocab_size4output=6000,
                embedding_dim=128,
                hidden_size=256,
                num_layers=2,
                bid=bid,
                net_category=rnn_type
            )

            src = randint(3, 5000, (3, 8))
            tgt = cat([
                full((3, 1), 1),
                randint(3, 6000, (3, 7)),
                full((3, 1), 2)
            ], dim=1)

            # 前向传播
            logits = model(src, tgt)
            assert logits.shape == (3, 8, 6000), f"Logits Error Size: {logits.shape}"

            # 生成测试
            generated = model.generate(src, max_len=10)
            assert generated.shape[0] == 3, f"Generation Batch Error: {generated.shape}"
            assert generated.shape[1] <= 10, f"Generation Length Error: {generated.shape}"

            print(f"Successfully!")
            print(f"- Logits Shape: {logits.shape}")
            print(f"- Generation Shape: {generated.shape}")
            starts()
            print()

        except Exception as e:
            print(f"Failed: {e}")
            import traceback

            traceback.print_exc()
