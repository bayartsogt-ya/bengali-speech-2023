import os
import re
import logging
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import pandas as pd

from .tokenizers import ALPHABET_LIST_OOD, dump_default_ood_vocab
from .datasets import (
    load_competition_train_data,
    load_competition_valid_data,
    load_madasr2023_train,
    load_madasr2023_dev,
    load_openslr_53
)

import torch
from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer, AutoConfig, AutoProcessor
import librosa
import soundfile as sf
from bnunicodenormalizer import Normalizer

DEFAULT_RATE = 16_000

# set up logging
logger = logging.getLogger(__name__)


def _clean_sentence_col(text: str):
    bnorm = Normalizer()
    keep_chars = "".join(ALPHABET_LIST_OOD) + " "
    normalized_list = [bnorm(word)["normalized"] for word in text.split()]
    text = " ".join([x for x in normalized_list if x])
    text = re.sub(f"[^{keep_chars}]", "", text)
    return text


def _clean_sentence_col_multip(df: pd.DataFrame, df_name: str):
    with Pool() as pool:
        df["sentence"] = list(tqdm(
            pool.imap(_clean_sentence_col, df.sentence.to_list()),
            total=len(df),
            desc=df_name,
        ))


def _write_desired_rate(row):
    """Given path and dataset name, write to resampled example to cache directory."""
    cache_dir, audio = row
    new_path = os.path.join(cache_dir, audio.split("/")[-1].split(".")[0] + ".wav")
    try:
        waveform, orig_sr = sf.read(audio)
        if len(waveform.shape) > 1:
            waveform = librosa.to_mono(waveform)

        if orig_sr != DEFAULT_RATE:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=DEFAULT_RATE)
            sf.write(new_path, waveform, DEFAULT_RATE)
            audio = new_path

    except KeyboardInterrupt as e:
        raise KeyboardInterrupt(e)
    except sf.LibsndfileError as e:
        logger.exception(e)
        return {
            "new_path": np.nan,
            "input_length": np.nan,
            "sampling_rate_orig": np.nan,
            "sampling_rate": np.nan,
        }


    return {
        "new_path": audio,
        "input_length": waveform.shape[0],
        "sampling_rate_orig": orig_sr,
        "sampling_rate": DEFAULT_RATE,
    }


def _write_desired_rate_multip(df: pd.DataFrame, cache_dir: str):
    output = []
    with Pool() as pool:
        output = list(tqdm(
            pool.imap(_write_desired_rate, [(cache_dir, audio) for audio in df.audio.to_list()]),
            total=len(df),
            desc=cache_dir,
        ))

    return pd.DataFrame(output)


def cache_and_preprocess_dataset(df: pd.DataFrame, name: str, cache_dir: str):
    _clean_sentence_col_multip(df, name)
    os.makedirs(cache_dir)
    tmp_df = _write_desired_rate_multip(df, cache_dir)

    # switch column names:
    df.rename({"audio":"audio_orig"}, axis=1, inplace=True)
    tmp_df.rename({"new_path":"audio"}, axis=1, inplace=True)

    df = pd.concat([df, tmp_df], axis=1)

    # extra features
    df["duration"] = df.input_length / df.sampling_rate
    df["wpm"] = df.apply(lambda x: 60 * len(x.sentence.split()) / x.duration, axis=1)

    df.to_csv(cache_dir + ".csv", index=False)
    return df


class DatasetInterface(torch.utils.data.Dataset):
    def __init__(
        self, 

        # augment_transforms,
        tokenizer: AutoTokenizer,
        feature_extractor: AutoFeatureExtractor,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.df = pd.DataFrame()

        return NotImplementedError()
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        waveform, sr = sf.read(row.audio)
        return {
            "input_values": self.feature_extractor(waveform, sampling_rate=sr).input_values[0],
            "labels": self.tokenizer(row.sentence).input_ids
        }


class TrainDataset(DatasetInterface):
    def __init__(
        self, 

        # paths
        path_to_competition_data: str = "./data/bengaliai-speech",
        path_to_madasr: str = "./data/madasr23dataset",
        valid_offset: int = 1480,

        # filtering
        min_sec = 0.,
        max_sec = 15.,

        min_wpm = 30.,
        max_wpm = 250.,

        # caching
        cache_dir="./cache/train",
        **kwargs,
    ):
        super().__init__(**kwargs)

        dataset_dict = {
            "comp_train": load_competition_train_data(path_to_competition_data),
            "comp_valid": load_competition_valid_data(path_to_competition_data)[valid_offset:].reset_index(drop=True),
            "madasr_train": load_madasr2023_train(path_to_madasr),
            "openslr53": load_openslr_53(),
        }

        for df in dataset_dict.values():
            assert list(df.columns) == ["audio", "sentence"]

        # clean sentence column
        for name in dataset_dict.keys():
            _cache_dir = os.path.join(cache_dir, name)
            if not os.path.isdir(_cache_dir):
                logger.info(f"Run preprocess function for {name}")
                _local_df = cache_and_preprocess_dataset(dataset_dict[name], name, _cache_dir).dropna()
            else:
                logger.info(f"Found cached output for {name}: {_cache_dir}.csv")
                _local_df = pd.read_csv(_cache_dir + ".csv").dropna()
            logger.info(f"Done reading: {name=} {_local_df.shape}")

            _local_df = _local_df[_local_df.duration.apply(lambda x: min_sec <= x <= max_sec)]
            logger.info(f"After applying duration filter {min_sec=} {max_sec=} => {_local_df.shape}")

            _local_df = _local_df[_local_df.wpm.apply(lambda x: min_wpm <= x <= max_wpm)]
            logger.info(f"After applying WPM filter {min_wpm=} {max_wpm=} => {_local_df.shape}")

            dataset_dict[name] = _local_df

        ## Weighing datasets
        weights_dict = {
            "comp_train": 1,
            "comp_valid": 10,
            "madasr_train": 1,
            "openslr53": 1,
        }

        logger.info(f"{weights_dict=}")

        list_df = []
        for dname in dataset_dict.keys():
            list_df += [dataset_dict[dname].dropna()] * weights_dict[dname]

        self.df = pd.concat(list_df)
        self.df = self.df.reset_index(drop=True)
        self.df = self.df[["audio", "sentence"]]

        logger.info(f"\n\n{self.df.shape=}")
        logger.info(self.df.head().to_markdown())


class ValidationDataset(DatasetInterface):
    def __init__(
        self, 

        # paths
        path_to_competition_data: str = "./data/bengaliai-speech",
        valid_offset: int = 1480,

        # 
        cache_dir="./cache/valid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        name = "comp_valid"
        _cache_dir = os.path.join(cache_dir, name)
        if not os.path.isdir(_cache_dir):
            logger.info(f"Run preprocess function for {name}")
            self.df = load_competition_valid_data(path_to_competition_data)[:valid_offset].reset_index(drop=True)
            self.df = cache_and_preprocess_dataset(self.df, name, _cache_dir).dropna()

        else:
            logger.info(f"Found cached output for {name}: {_cache_dir}.csv")
            self.df = pd.read_csv(_cache_dir + ".csv").dropna()
            logger.info(f"Done reading: {name=} {self.df.shape}")

        self.df = self.df.reset_index(drop=True)
        self.df = self.df[["audio", "sentence"]]

        logger.info(f"\n{self.df.shape=}")
        logger.info(self.df.head().to_markdown())
