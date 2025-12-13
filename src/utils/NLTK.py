#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/9 20:25
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   NLTK.py
# @Desc     :   

from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu


class NLTKTokenizer:
    """ NLTK Tokeniser """

    def __init__(self, lang: str = "en", INV: bool = False) -> None:
        self._lang = lang
        self._INV = INV
        self._tokenizer = None

    def __enter__(self):
        if self._INV:
            self._tokenizer = TreebankWordDetokenizer()
        else:
            self._tokenizer = TreebankWordTokenizer()

        # print(f"NLTK Tokenizer for language '{self._lang}' initialized.")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tokenizer:
            self._tokenizer = None
            # print(f"NLTK Tokenizer for language '{self._lang}' closed.")

    def tokenize(self, text: str) -> list[str]:
        """ Tokenize the input text """
        return self._tokenizer.tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        """ Detokenize the input tokens """
        return self._tokenizer.detokenize(tokens)


def bleu_score(reference: list[str], candidate: list[str], smooth: bool = True) -> float:
    """ Calculate BLEU score for a single candidate against a reference
    :param reference: reference text
    :param candidate: candidate text
    :param smooth: whether to apply smoothing
    :return: BLEU score
    """
    if smooth:
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference], candidate, smoothing_function=smoothie)
    else:
        return sentence_bleu([reference], candidate)


if __name__ == "__main__":
    sample_text = "Hello, world! This is a test."

    with NLTKTokenizer(lang="en") as tokenizer:
        tokens = tokenizer.tokenize(sample_text)
        print("Tokens:", tokens)

    with NLTKTokenizer(lang="en", INV=True) as detokenizer:
        detext = detokenizer.detokenize(tokens)
        print("Detokenized Text:", detext)
