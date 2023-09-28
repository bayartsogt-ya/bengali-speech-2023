import json
from transformers import Wav2Vec2CTCTokenizer


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


def write_vocab_file(vocab_file_path: str = "./data/vocab.json"):
    characters = ["<unk>", "<pad>", "|"] + PUNCTUATION_SYMBOLS + ALPHABET_LIST
    vocab = {x: i for i, x in enumerate(characters)}
    with open(vocab_file_path, 'w') as vocab_file:
        json.dump(vocab, vocab_file, ensure_ascii=False)


def get_default_wav2vec_tokenizer(vocab_file: str = "./data/vocab.json"):
    # https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer
    return Wav2Vec2CTCTokenizer(vocab_file=vocab_file)
