"""Entry point for training the model."""

import shutil
import json
import os
import re
import argparse
import logging

import numpy as np

from bengali_speech.datasets import read_bengaliai_speech_2023_using_hf_datasets
from bengali_speech.tokenizers import dump_default_ood_vocab
from bengali_speech.data_collators import DataCollatorCTCWithPadding
from bengali_speech.evaluate import get_compute_metrics_func
from bengali_speech.kaggle import upload_to_kaggle
from bengali_speech.utils import log_title_with_multiple_lines


from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer, AutoConfig, AutoProcessor
from transformers import Trainer
from transformers import TrainingArguments

from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets, DatasetDict
from datasets.utils.info_utils import VerificationMode
from bnunicodenormalizer import Normalizer

from audiomentations import AddBackgroundNoise, PolarityInversion, Compose, OneOf, RoomSimulator, AddGaussianNoise, TimeStretch, Normalize


DEFAULT_RATE = 16_000

# set up logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # create a parser for variables under parameters and transformers.TrainingArguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="patrickvonplaten/tiny-wav2vec2-no-tokenizer")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--experiment_number", type=int, default=0)
    parser.add_argument("--wav2vec_freeze_feature_extractor", action="store_true")

    # hf training args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=240_000)
    parser.add_argument("--dataloader_num_workers", type=int, default=12)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--to_kaggle", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # preprocessing args
    parser.add_argument("--min_sec", type=float, default=2.)
    parser.add_argument("--max_sec", type=float, default=10.)
    parser.add_argument("--num_shards_train", type=int, default=None)
    parser.add_argument("--shard_index_train", type=int, default=0)
    parser.add_argument("--num_shards_validation", type=int, default=None)
    parser.add_argument("--shard_index_validation", type=int, default=0)
    args = parser.parse_args()

    # print args nicely
    log_title_with_multiple_lines("Arguments:")
    for arg in vars(args):
        logger.info("%s: %s", arg, getattr(args, arg))

    experiment_name = f"bengali-2023-{args.experiment_number:04d}"

    training_args = TrainingArguments(
        output_dir=os.path.join("output", experiment_name),
        # group_by_length=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        
        # LR
        learning_rate=3e-5,
        # weight_decay=0.005,
        # warmup_ratio=0.1,
        warmup_steps=14000,

        # EVAL & SAVE
        logging_steps=200,
        
        evaluation_strategy="steps",
        eval_steps=2000,  # means that every 16k examples will be evaluated

        save_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
        
        fp16=args.fp16,

        load_best_model_at_end=True,
        log_level="debug",

        lr_scheduler_type="cosine",

        # report
        push_to_hub=args.push_to_hub,
        metric_for_best_model="eval_validation_wer",
        greater_is_better=False,
        report_to=["tensorboard", "wandb"],
        # report_to="none",
        run_name=experiment_name,
    )

    ######################### ALL HUGGINGFACE CONFIG FIRST #########################
    log_title_with_multiple_lines("Loading hugginface config, tokenizer, and feature extractor.")
    # load tokenizer
    tokenizer_name = args.tokenizer_name
    tokenizer_kwargs = {}
    if tokenizer_name is None:
        tokenizer_name = training_args.output_dir
        tokenizer_kwargs = dump_default_ood_vocab(training_args.output_dir, args.base_model_name)

    # load config, tokenizer, and feature extractor
    config = AutoConfig.from_pretrained(args.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model_name)

    config.update(
        {
            "ctc_loss_reduction": "mean",
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
        }
    )

    # save to output dir
    config.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    feature_extractor.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # load model
    model = AutoModelForCTC.from_pretrained(
        args.base_model_name,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )

    if hasattr(model, "freeze_feature_encoder") and args.wav2vec_freeze_feature_extractor:
        logger.info("Freezing a feature extractor.")
        model.freeze_feature_encoder()

    ######################### LOAD DATA #########################
    cache_dir = os.path.join("cache", "bengaliai-speech-cv13")
    ##################### <-------------------------
    if not os.path.isdir(cache_dir):

        log_title_with_multiple_lines("Preprocessing data and write it to disk.")

        # ################### COMPETITION DATA ###################
        # dataset = read_bengaliai_speech_2023_using_hf_datasets(path_to_data="data/bengaliai-speech")
        # dataset = dataset.cast_column("audio", Audio(sampling_rate=DEFAULT_RATE))

        # logger.critical("Using only validation data")
        # dataset = DatasetDict({
        #     "train": concatenate_datasets([
        #         dataset["validation"].shard(num_shards=args.num_shards_validation, index=index)
        #         for index in range(args.num_shards_validation) if index != args.shard_index_validation
        #     ]),
        #     "validation": dataset["validation"].shard(num_shards=args.num_shards_validation, index=args.shard_index_validation),
        #     "example": dataset["example"],
        # })

        # ################## SLR53 DATA ###################
        # dataset = load_dataset("openslr", "SLR53")
        # dataset = dataset.remove_columns(["path"])
        # dataset = dataset.cast_column("audio", Audio(sampling_rate=DEFAULT_RATE))
        # logger.info(dataset)

        # def filter_openslr53_by_word_len(batch):
        #     # found some error 
        #     return len(batch["sentence"].split()) < 20
        # logger.info("After filter_openslr53_by_word_len:")
        # dataset["train"] = dataset["train"].filter(filter_openslr53_by_word_len, num_proc=args.num_proc)
        # logger.info(dataset)

        ################## OPENCV ###################
        dataset = DatasetDict({
            "train": load_dataset("mozilla-foundation/common_voice_13_0", "bn", split="train+validation+test")
        })
        dataset = dataset.cast_column("audio", Audio(sampling_rate=DEFAULT_RATE))
        logger.info("Initial:")
        logger.info(dataset)
        logger.info(dataset["train"][0])

        def filter_by_upvote(batch):
            return  batch["up_votes"] > batch["down_votes"]
        dataset = dataset.filter(filter_by_upvote, num_proc=args.num_proc, writer_batch_size=200)
        logger.info("After Upvote filtering:")
        logger.info(dataset)

        ######################### DATA PREPROCESSING #########################
        keep_chars = "".join(x for x in tokenizer.vocab if x not in ["[UNK]", "[PAD]"])
        logger.critical(f"Keep only following characters: {keep_chars}, Vocab size: {len(tokenizer.vocab)}")

        def clean_text(batch, bnorm: Normalizer):
            normalized_list = [bnorm(word)["normalized"] for word in batch["sentence"].split()]
            batch["sentence"] = "|".join([x for x in normalized_list if x])
            batch["sentence"] = re.sub(f"[^{keep_chars}]", "", batch["sentence"])
            return batch

        dataset = dataset.map(clean_text, num_proc=args.num_proc, fn_kwargs={"bnorm": Normalizer()})
        logger.info("After cleaning:")
        logger.info(dataset)
        logger.info(f"{dataset['train'][:3]['sentence']}")

        def filter_by_length(batch):
            duration = batch["audio"]["array"].shape[0] / batch["audio"]["sampling_rate"]
            return args.min_sec < duration < args.max_sec

        # if args.min_sec and args.max_sec:
        dataset["train"] = dataset["train"].filter(filter_by_length, num_proc=args.num_proc)
        logger.info("After filter_by_length:")
        logger.info(dataset)

        def prepare_dataset(batch):
            batch["input_values"] = batch["audio"]["array"]
            batch["sampling_rate"] = batch["audio"]["sampling_rate"]
            batch["labels"] = tokenizer(batch["sentence"]).input_ids

            # stat related
            batch["input_length"] = len(batch["input_values"])
            batch["duration"] = batch["input_length"] / batch["sampling_rate"]
            batch["num_words"] = len(batch["sentence"].split("|"))
            batch["word_per_minute"] = 60 * batch["num_words"] / batch["duration"]

            return batch

        dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=args.num_proc, writer_batch_size=100)
        logger.info(dataset)
        for k, v in dataset['train'][0].items():
            if k == "input_values": continue
            logger.info(f"{k}={v}")
        logger.info("Done preparing dataset.")

        # only saving the train
        dataset["train"].save_to_disk(cache_dir, num_proc=args.num_proc)
        # dataset.save_to_disk(cache_dir, num_proc=args.num_proc)
        raise Exception(f"Success writing to disk `{cache_dir=}`")
    # ##################### <-------------------------

    log_title_with_multiple_lines(f"Directly reading from cache {cache_dir}")
    dataset = load_from_disk("./cache/bengaliai-speech-validation")
    dataset_openslr_53 = load_from_disk("./cache/bengaliai-speech-openslr-53")
    dataset_cv13_train_valid_test = load_from_disk("./cache/bengaliai-speech-cv13")
    logger.info(f"{dataset=}")
    logger.info(f"{dataset_openslr_53=}")
    logger.info(f"{dataset_cv13_train_valid_test=}")


    dataset["train"] = concatenate_datasets([dataset["train"], dataset_openslr_53, dataset_cv13_train_valid_test])
    logger.info(dataset)
    logger.info(f"{dataset['train'][0]['input_length']} \n {dataset['train'][0]['labels']}")

    np.seterr('raise')

    SAMPLE_RATE = 16000

    augment_composer = Compose([
        # # normalize for 
        # Normalize(p=1.),

        # time stretch
        TimeStretch(
            min_rate=1.0,
            max_rate=1.7,
            leave_length_unchanged=False,
            p=0.1,
        ),

        # background
        OneOf([
            AddBackgroundNoise(
                sounds_path="./data/audiomentation_input/BollywoodMusic_16k/",
                min_snr_in_db=3.0,
                max_snr_in_db=30.0,
                noise_rms = "relative",
                noise_transform=PolarityInversion(),
                p=0.4,
                lru_cache_size=10,
            ),
            AddBackgroundNoise(
                sounds_path="./data/audiomentation_input/ThemeMusic_16k/",
                min_snr_in_db=3.0,
                max_snr_in_db=30.0,
                noise_rms = "relative",
                noise_transform=PolarityInversion(),
                p=0.4,
                lru_cache_size=10,
            ),
            AddBackgroundNoise(
                sounds_path="./data/audiomentation_input/Applause_16k/",
                min_snr_in_db=3.0,
                max_snr_in_db=30.0,
                noise_rms = "relative",
                noise_transform=PolarityInversion(),
                p=0.1,
                lru_cache_size=10,
            ),
            RoomSimulator(
                min_absorption_value=0.05,
                max_absorption_value=0.1,
                p=0.5,
            ),
            AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.015,
                p=0.5
            )
        ], p=0.5)
    ])

    def augmentation_transform(batch):
        """This function will be called after .forward input check.
        So only `input_values`, `labels` will be present
        """
        def apply_composer(waveform):
            try:
                return augment_composer(np.array(waveform, dtype=np.float32), 16000)
            except KeyboardInterrupt as e:
                raise KeyboardInterrupt(e)
            except Exception as e:
                # with open('error_occurred.npy', 'wb') as f:
                #     np.save(f, waveform)

                # logger.error(f"""
                #     ---> Error occurred in `apply_composer` 
                #     INPUT={np.array(waveform, dtype=np.float32)} 
                #     ERROR={e}
                # """)
                # logger.error("-----------------------------")
                # logger.error("Reproduce the error:")
                # logger.error(f"{np.square(waveform)=}")
                # logger.error(f"{np.mean(np.square(waveform))=}")
                # logger.error(f"{np.sqrt(np.mean(np.square(waveform)))=}")
                # logger.error("-----------------------------")
                # logger.error("Reproduce the error:")

                # raise Exception(e)
                return waveform
        
        augmented = [apply_composer(waveform) for waveform in batch["input_values"]]

        batch["input_values"] = [
            feature_extractor(waveform, sampling_rate=16000).input_values[0]
            for waveform in augmented
            if waveform is not None
        ]

        return batch

    dataset["train"].set_transform(augmentation_transform)

    def _prepare_dataset(batch):
        batch["input_values"] = feature_extractor(batch["input_values"], sampling_rate=batch["sampling_rate"]).input_values[0]
        return batch

    dataset["validation"] = dataset["validation"].map(_prepare_dataset, num_proc=args.num_proc, writer_batch_size=100)
    dataset["example"] = dataset["example"].map(_prepare_dataset, num_proc=args.num_proc, writer_batch_size=100)
    logger.info(dataset)
    logger.info(f'{dataset["train"][0].keys()=}')
    logger.info(f'{dataset["validation"][0].keys()=}')
    logger.info(f'{dataset["example"][0].keys()=}')
    logger.info("Done preparing dataset.")

    ######################### LOAD MODEL AND TRAIN #########################
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=get_compute_metrics_func(processor=processor),
        train_dataset=dataset["train"],
        eval_dataset={
            "validation": dataset["validation"],
            "example": dataset["example"],
        },
        tokenizer=processor.feature_extractor,
    )


    logger.info(f"coping train scripts to {training_args.output_dir}")
    shutil.copy("./train.py", training_args.output_dir)
    shutil.copy("./run_train.sh", training_args.output_dir)

    log_title_with_multiple_lines("Start training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPTED! Starting evaluation with current state')
        trainer.is_in_train = False

    trainer.save_model()
    trainer.save_state()
    log_title_with_multiple_lines("Done Training and Uploading Output")

    if args.to_kaggle:
        upload_to_kaggle(experiment_name)

    log_title_with_multiple_lines("Done Uploading Output to Kaggle")
