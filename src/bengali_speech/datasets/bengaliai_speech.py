import time
import logging
import pandas as pd
from datasets import Dataset, DatasetDict, Audio, Value, load_dataset


logger = logging.getLogger(__name__)

DEFAULT_RATE = 16_000

def read_bengaliai_speech_2023(path_to_data: str) -> DatasetDict:
    st = time.time()
    logger.info("Reading Bengali Speech 2023 competition data...")
    df = pd.read_csv(f"{path_to_data}/train.csv")

    df_train = df[df.split == "train"].reset_index(drop=True)
    df_validation = df[df.split != "train"].reset_index(drop=True)

    df_train["audio"] = df_train.id.apply(lambda x: f"{path_to_data}/train_mp3s/{x}.mp3")
    df_validation["audio"] = df_validation.id.apply(lambda x: f"{path_to_data}/train_mp3s/{x}.mp3")

    df_example = pd.read_csv(f"{path_to_data}/annoated.csv",sep="\t")
    df_example = df_example.rename({"file": "id"}, axis=1)
    df_example["audio"] = df_example.id.apply(lambda x: f"{path_to_data}/examples/{x}")

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train[["id", "audio", "sentence"]]),
        "validation": Dataset.from_pandas(df_validation[["id", "audio", "sentence"]]),
        "example": Dataset.from_pandas(df_example[["id", "audio", "sentence"]]),
    })

    dataset = dataset.cast_column("audio", Audio(sampling_rate=DEFAULT_RATE))
    dataset = dataset.cast_column("id", Value("string"))
    dataset = dataset.cast_column("sentence", Value("string"))
    logger.info(f"Directory to cache: {dataset.cache_files}")
    logger.info(f"Train dataset column features: {dataset['train'].features}")
    logger.info(f"Done reading Bengali Speech 2023 competition data in {time.time() - st:.2f}s")

    return dataset



def read_bengaliai_speech_2023_using_hf_datasets(path_to_data: str) -> DatasetDict:
    st = time.time()
    logger.info("Reading Bengali Speech 2023 competition data...")

    ds = load_dataset("csv", data_files=[f"{path_to_data}/train.csv"], split="train")
    ds_example = load_dataset("csv", data_files=[f"{path_to_data}/annoated.csv"], split="train", delimiter="\t")
    ds_example = ds_example.rename_column("file", "id")

    def _get_train_audio_path(batch, subfolder, add_mp3):
        batch["audio"] = f"{path_to_data}/{subfolder}/{batch['id']}"
        if add_mp3:
            batch["audio"] += ".mp3"
        return batch
    
    ds = ds.map(_get_train_audio_path, fn_kwargs={"subfolder": "train_mp3s", "add_mp3": True})
    ds_example = ds_example.map(_get_train_audio_path, fn_kwargs={"subfolder": "examples", "add_mp3": False})

    ds_train = ds.filter(lambda x: x["split"] == "train")
    ds_valid = ds.filter(lambda x: x["split"] == "valid")

    # SLICE THE TRAIN
    ds_train = ds_train.select(range(int(0.3 * len(ds_train))))

    ds_train = ds_train.remove_columns(["split"])
    ds_valid = ds_valid.remove_columns(["split"])

    dataset = DatasetDict({
        "train": ds_train,
        "validation": ds_valid,
        "example": ds_example,
    })

    dataset = dataset.cast_column("audio", Audio(sampling_rate=DEFAULT_RATE))

    logger.info(f"Done reading Bengali Speech 2023 competition data in {time.time() - st:.2f}s")

    return dataset
