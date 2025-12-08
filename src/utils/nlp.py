#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 22:33
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   nlp.py
# @Desc     :   

from collections import Counter
from pathlib import Path
from re import compile, sub
from pandas import DataFrame
from spacy import load
from tqdm import tqdm

from src.configs.cfg_base import CONFIG
from src.utils.decorator import timer

spacy4cn = None
spacy4en = None


# @timer
def regular_chinese(words: list[str]) -> list[str]:
    """ Retain only Chinese characters in the list of words
    :param words: list of words to process
    :return: list of words containing only Chinese characters
    """
    pattern = compile(r"[\u4e00-\u9fa5]+")

    chinese = [word for word in words if pattern.fullmatch(word)]

    # print(f"Retained {len(chinese)} Chinese words from the original {len(words)} words.")

    return chinese


@timer
def regular_english(words: list[str]) -> list[str]:
    """ Retain only English characters in the list of words
    :param words: list of words to process
    :return: list of words containing only English characters
    """
    pattern = compile(r"^[A-Za-z]+$")

    english: list[str] = [word.lower() for word in words if pattern.fullmatch(word)]

    print(f"Retained {len(english)} English words from the original {len(words)} words.")

    return english


@timer
def count_frequency(words: list[str], top_k: int = 10, freq_threshold: int = 3) -> tuple[list, DataFrame]:
    """ Get frequency of Chinese words
    :param words: list of words to process
    :param top_k: number of top frequent words to return
    :param freq_threshold: frequency threshold to separate high and low frequency words
    :return: DataFrame containing words and their frequencies
    """
    # Get word frequency using Counter
    counter = Counter(words)
    words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    high_freq: list[str] = [word for word, count in words if count > freq_threshold]
    low_freq: list[str] = [word for word, count in words if count <= freq_threshold]

    cols: list[str] = ["word", "frequency"]
    sorted_freq = words[:top_k]
    df: DataFrame = DataFrame(sorted_freq, columns=cols)
    sorted_df = df.sort_values(by="frequency", ascending=False)

    print(f"Word Frequency Results:\n{sorted_df}")
    print(f"{len(low_freq)}, {len(low_freq) / len(counter):.2%} low frequency words has been filtered.")

    return high_freq, sorted_df


@timer
def unique_characters(content: str) -> list[str]:
    """ Get unique words from the list
    :param content: text content to process
    :return: list of unique words
    """
    chars: list[str] = list(content)
    # Get unique words by converting the list to a set and back to a sorted list
    # - sort based on Unicode code point order
    unique: list[str] = list(sorted(set(chars)))

    print(f"Extracted {len(unique)} unique words from the original {len(chars)} words.")

    return unique


@timer
def extract_zh_chars(filepath: str | Path, pattern: str = r"[^\u4e00-\u9fa5]") -> tuple[list, list]:
    """ Get Chinese characters from the text content
    :param filepath: path to the text file
    :param pattern: regex pattern to remove unwanted characters
    :return: list of Chinese characters
    """
    chars: list[str] = []
    lines: list[str] = []
    with open(str(filepath), "r", encoding="utf-8") as file:
        for line in file:
            line = sub(pattern, "", line).strip()
            if not line:
                continue
            lines.append(line)
            for word in list(line):
                chars.append(word)

    print(f"Total number of Chinese characters: {len(chars)}")
    print(f"Total number of lines in the Chinese content: {len(lines)}")

    return chars, lines


@timer
def spacy_single_tokeniser(content: str, lang: str, strict: bool = False) -> list[str]:
    """ SpaCy NLP Processor for an English or a Chinese text
    :param content: a text content to process
    :param lang: language code for the text (e.g., 'en' for English, 'cn' for Chinese)
    :param strict: whether to enforce strict token filtering (default is False)
    :return: list of tokens
    """
    nlp = None

    global spacy4cn, spacy4en
    match lang:
        case "cn":
            if spacy4cn is None:
                spacy4cn = load(CONFIG.FILEPATHS.SPACY_MODEL_CN)
                print("SpaCy Chinese Model initialized.")
                print(f"{spacy4cn.pipe_names} loaded.")
            nlp = spacy4cn
        case "en":
            if spacy4en is None:
                spacy4en = load(CONFIG.FILEPATHS.SPACY_MODEL_EN)
                print(f"SpaCy English Model initialized.")
                print(f"{spacy4en.pipe_names} loaded.")
            nlp = spacy4en
        case _:
            raise ValueError(f"Unsupported language: {lang}")

    words: list[str] = []
    doc = nlp(content)

    match lang:
        case "cn":
            if strict:
                words = [
                    token.text for token in doc
                    if token.text.strip()
                       and not token.is_stop
                       and not token.is_punct
                       and any(c.isalnum() for c in token.text)
                ]
            else:
                words = [
                    token.text
                    for token in doc
                    if token.text.strip()
                       and not token.is_stop
                       and any(c.isalnum() for c in token.text)
                ]
        case "en":
            if strict:
                words = [
                    token.lemma_.lower() for token in doc
                    if not token.is_stop
                       and token.text.strip()
                       and len(token.lemma_) > 1
                       and any(c.isalnum() for c in token.text)
                ]
            else:
                words = [
                    token.lemma_.lower() for token in doc
                    if not token.is_stop and token.text.strip() and len(token.lemma_) > 1
                ]
        case _:
            raise ValueError(f"Unsupported language: {lang}")

    print(f"The {len(words)} words has been segmented using SpaCy Tokeniser.")

    return words


@timer
def spacy_batch_tokeniser(
        contents: list[str], lang: str = "en", batches: int = 100, strict: bool = False
) -> list[list[str]]:
    """ SpaCy NLP Processor for a batch of English texts
    :param contents: list of text contents to process
    :param lang: language code for the texts (default is 'en' for English)
    :param batches: number of texts to process in each batch
    :param strict: whether to enforce strict token filtering (default is False)
    :return: list of tokenized texts
    """
    nlp = None

    global spacy4cn, spacy4en
    match lang:
        case "cn":
            if spacy4cn is None:
                spacy4cn = load(CONFIG.FILEPATHS.SPACY_MODEL_CN)
                print("SpaCy Chinese Model initialized.")
                print(f"{spacy4cn.pipe_names} loaded.")
            nlp = spacy4cn
        case "en":
            if spacy4en is None:
                spacy4en = load(CONFIG.FILEPATHS.SPACY_MODEL_EN)
                print(f"SpaCy English Model initialized.")
                print(f"{spacy4en.pipe_names} loaded.")
            nlp = spacy4en
        case _:
            raise ValueError(f"Unsupported language: {lang}")

    words: list[list[str]] = []
    tokens: list[str] = []
    for doc in tqdm(nlp.pipe(contents, batch_size=batches), total=len(contents), desc="SpaCy Batch Tokeniser"):
        match lang:
            case "cn":
                if strict:
                    tokens = [
                        token.text for token in doc
                        if token.text.strip()
                           and not token.is_stop
                           and not token.is_punct
                           and any(c.isalnum() for c in token.text)
                    ]
                else:
                    tokens = [
                        token.text
                        for token in doc
                        if token.text.strip()
                           and not token.is_stop
                           and any(c.isalnum() for c in token.text)
                    ]
            case "en":
                if strict:
                    tokens = [
                        token.lemma_.lower() for token in doc
                        if not token.is_stop
                           and token.text.strip()
                           and len(token.lemma_) > 1
                           and any(c.isalnum() for c in token.text)
                    ]
                else:
                    tokens = [
                        token.lemma_.lower() for token in doc
                        if not token.is_stop and token.text.strip() and len(token.lemma_) > 1
                    ]
            case _:
                raise ValueError(f"Unsupported language: {lang}")
        words.append(tokens)

    print(f"Average length is {sum(len(w) for w in words) / len(words):.2f} words per content.")

    return words


def build_word2id_seqs(
        contents: list[list[str]], dictionary: dict[str, int],
        UNK: str = "<UNK>"
) -> list[list[int]]:
    """ Build word2id sequences from contents using the provided dictionary
    :param contents: list of texts to convert
    :param dictionary: word2id mapping dictionary
    :return: list of word2id sequences
    """
    sequences: list[list[int]] = []
    for content in contents:
        sequence: list[int] = []
        for word in content:
            if word in dictionary:
                sequence.append(dictionary[word])
            else:
                sequence.append(dictionary[UNK])
        sequences.append(sequence)

    return sequences


@timer
def check_vocab_coverage(words: list[str], dictionary: dict[str, int]) -> float:
    """ Check the vocab coverage
    :param words: list of words
    :param dictionary: word2id mapping dictionary
    :return: vocab coverage
    """
    counter: int = sum(1 for word in words if word in dictionary)
    coverage: float = counter / len(words)

    if coverage >= 0.95:
        rating = "Perfect"
    elif coverage >= 0.90:
        rating = "Good"
    elif coverage >= 0.85:
        rating = "Enough"
    else:
        rating = "Bad"

    print(f"The coverage of vocabs in the sentences is {coverage:.2%}, and {rating}.")

    return coverage


if __name__ == "__main__":
    pass
