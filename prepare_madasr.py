import os
import pandas as pd
from glob import glob
from tqdm.auto import tqdm

# read actual data
def get_data(path):
    data = []
    with open(os.path.join(path, "text"), "r") as fp:
        for line in fp.readlines():
            _id, *tokens = line.split()
            spk, text, utt =  _id.split("_")
            data.append({
                "_id": _id, "spk": spk, "text": text, "utt": utt, "sentence": " ".join(tokens), "num_words": len(tokens)
            })
    durations = []
    with open(os.path.join(path, "utt2dur"), "r") as fp:
        for line in fp.readlines():
            _id, duration = line.split()
            durations.append({
                "_id": _id, "duration": float(duration)
            })


    df = pd.DataFrame(data)
    df_duration = pd.DataFrame(durations)
    df = df.merge(df_duration)

    ####### stats
    # word per minute:
    df["wpm"] = 60 * df.num_words / df.duration

    return df

path_to_corpus = "./data/madasr23dataset/RESPIN_ASRU_Challenge_2023/corpus/bn"
path_to_wav_files = "./data/madasr23dataset/*/*.wav"
path_to_output = "./data/madasr23dataset"

df_dev = get_data(os.path.join(path_to_corpus, "dev"))
df_train = get_data(os.path.join(path_to_corpus, "train"))

print("loaded corpus")

train_path_list = glob(path_to_wav_files)
data = []
for path in train_path_list:
    utt = path.split("/")[-1].split(".")[0]
    data.append({
        "utt": utt, "audio": path,
    })

df_path_utt = pd.DataFrame(data)
df_path_utt = df_path_utt.drop_duplicates("utt")  # we assume `utt` has some duplicates

df_train_with_path = df_train.merge(df_path_utt, how="inner", on="utt")
df_dev_with_path = df_dev.merge(df_path_utt, how="inner", on="utt")

print(df_train_with_path.shape, df_dev_with_path.shape)

cols_for_output = ["audio", "sentence"]
df_train_with_path[cols_for_output].to_csv(os.path.join(path_to_output, "train.tsv"), sep="\t", index=False)
df_dev_with_path[cols_for_output].to_csv(os.path.join(path_to_output, "dev.tsv"), sep="\t", index=False)
