import time
import logging
import pandas as pd
from datasets import Dataset, DatasetDict, Audio


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
    logger.info(f"Done reading Bengali Speech 2023 competition data in {time.time() - st:.2f}s")

    return dataset
