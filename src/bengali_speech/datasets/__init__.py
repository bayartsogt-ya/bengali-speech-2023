from .bengaliai_speech import *
import time
import os
import pandas as pd
from datasets import load_dataset

MAX_RECORDING_PER_USER = 200


def load_competition_valid_data(path_to_data):
    st = time.time()
    df = pd.read_csv(os.path.join(path_to_data, "train.csv"))
    df = df[df.split == "valid"].reset_index(drop=True)

    # construct audio path
    df["audio"] = df.id.apply(lambda x: os.path.join(path_to_data, "train_mp3s", f"{x}.mp3"))
    print(f"loaded competition data in {time.time() - st:.2f}s")
    return df[["audio", "sentence"]].reset_index(drop=True)


def load_competition_example_data(path_to_data):
    st = time.time()
    df = pd.read_csv(os.path.join(path_to_data, "annoated.csv"), sep="\t")
    df = df.rename({"file": "id"}, axis=1)

    # construct audio path
    df["audio"] = df.id.apply(lambda x: os.path.join(path_to_data, "examples", f"{x}"))
    print(f"loaded competition data in {time.time() - st:.2f}s")
    return df[["audio", "sentence"]].reset_index(drop=True)


def load_competition_train_data(path_to_data):
    st = time.time()
    df_meta = pd.read_csv(os.path.join(path_to_data, "train_metadata.csv"))
    df_nisqa = pd.read_csv(os.path.join(path_to_data, "NISQA_wavfiles.csv"))

    print(f"loaded meta data in {time.time() - st:.2f}s")

    df_nisqa["id"] = df_nisqa.deg.apply(lambda x: x.split(".")[0])
    df_nisqa = df_nisqa.drop(["deg", "model"], axis=1)
    df_all = df_meta.merge(df_nisqa, how="left", on="id")

    # filter out too high CER or NA or low MOS.
    df_filtered = df_all.query("ggl_cer < 0.8 & ykg_cer < 0.8 & not google_preds.isna() & mos_pred > 2.")

    # let's limit number of examples from a single user
    def get_limited_ids(row):
        id_list = list(row.id)
        mos_list = list(row.mos_pred)
        
        sorted_mos = sorted([(-x, i) for i, x in enumerate(mos_list)])[:MAX_RECORDING_PER_USER]
        return [id_list[i] for _, i in sorted_mos]

    _df_filtered = df_filtered.groupby("client_id")[["id", "mos_pred"]].apply(get_limited_ids)

    all_ids = []
    for _ids in _df_filtered:
        all_ids += _ids

    df_filtered_client_limited = pd.DataFrame({"id": all_ids})
    df_filtered_client_limited = df_filtered_client_limited.merge(df_all, on="id", how="left")

    # construct audio path
    df_filtered_client_limited["audio"] = df_filtered_client_limited.id.apply(
        lambda x: os.path.join(path_to_data, "train_mp3s", f"{x}.mp3"))

    print(f"{df_filtered_client_limited.shape=}")
    print(f"loaded competition data in {time.time() - st:.2f}s")
    return df_filtered_client_limited[["audio", "sentence"]].reset_index(drop=True)

def load_madasr2023_train(path_to_data):
    st = time.time()
    logger.info("Reading MADASR2023 competition train data...")
    df_train = pd.read_csv(os.path.join(path_to_data, "train.tsv"), sep="\t")
    logger.info(df_train.shape)
    logger.info(f"Done reading MADASR2023 competition train data in {time.time() - st:.2f}s")
    return df_train

def load_madasr2023_dev(path_to_data):
    st = time.time()
    logger.info("Reading MADASR2023 competition dev data...")
    df_dev = pd.read_csv(os.path.join(path_to_data, "dev.tsv"), sep="\t")
    logger.info(df_dev.shape)
    logger.info(f"Done reading MADASR2023 competition dev data in {time.time() - st:.2f}s")
    return df_dev


def load_openslr_53():
    st = time.time()
    logger.info("Reading OpenSLR-53...")
    dataset = load_dataset("openslr", "SLR53", split="train")
    dataset = dataset.remove_columns(["audio"])

    df = dataset.to_pandas()
    df = df.rename({"path": "audio"}, axis=1)

    # remove wrong labels
    df = df[df.sentence.apply(lambda x: len(x.split()) < 20)]

    logger.info(df.shape)
    logger.info(f"Done reading OpenSLR-53 data in {time.time() - st:.2f}s")
    return df.reset_index(drop=True)
