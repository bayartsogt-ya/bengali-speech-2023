import os
import json
import logging
from transformers import AutoConfig


logger = logging.getLogger(__name__)

"""
Bengali alphabet list is taken from:
https://huggingface.co/arijitx/wav2vec2-xls-r-300m-bengali/blob/main/alphabet.json
"""
ALPHABET_LIST = [
    "।", "ঁ", "ং", "ঃ", "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
    "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ", "ণ",
    "ত", "থ", "দ", "ধ", "ন", "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ",
    "স", "হ", "়", "া", "ি", "ী", "ু", "ূ", "ৃ", "ে", "ৈ", "ো", "ৌ", "্", "ৎ", "ৗ",
    "ড়", "ঢ়", "য়", "০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "ৰ"
]

PUNCTUATION_SYMBOLS = [
    ",", "!", "?", "."
]


# competition only alphabet
ALPHABET_LIST_OOD = [
    "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ", "ক", "খ",
    "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ", "ণ",
    "ত", "থ", "দ", "ধ", "ন", "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল",
    "শ", "ষ", "স", "হ", "ৎ", "ড়", "ঢ়", "য়"
]

PUNCTUATION_SYMBOLS = [
    " ", "!", "\"", "'", ",", "-", ".", ":", ";", "?", "।", "—", "”"
]


def write_vocab_file(vocab_file_path: str = "./data/vocab.json"):
    characters = ["<unk>", "<pad>", "|"] + PUNCTUATION_SYMBOLS + ALPHABET_LIST
    vocab = {x: i for i, x in enumerate(characters)}
    with open(vocab_file_path, 'w') as vocab_file:
        json.dump(vocab, vocab_file, ensure_ascii=False)


def dump_default_ood_vocab(output_dir, base_model_name) -> dict:
    """Writes a default Out of Distribution (OOD) Bengali as a vocab.json."""
    logger.info("Creating a new tokenizer.")
    vocab_file = os.path.join(output_dir, "vocab.json")
    os.makedirs(output_dir, exist_ok=True)
    vocab_dict = {x: i for i, x in enumerate(ALPHABET_LIST_OOD)}
    vocab_dict["|"] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    # save vocab dict to be loaded into tokenizer
    with open(vocab_file, "w") as file:
        json.dump(vocab_dict, file, ensure_ascii=False)

    config = AutoConfig.from_pretrained(base_model_name)
    return {
        "config": config if config.tokenizer_class is not None else None,
        "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "word_delimiter_token": "|",
    }
