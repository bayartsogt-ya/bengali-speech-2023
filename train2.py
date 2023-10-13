"""Entry point for training the model."""

import shutil
import json
import os
import re
import argparse
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer, AutoConfig, AutoProcessor
from transformers import Trainer
from transformers import TrainingArguments
from transformers import pipeline

from datasets import load_dataset, Audio, load_from_disk, concatenate_datasets, DatasetDict

from audiomentations import AddBackgroundNoise, PolarityInversion, Compose, OneOf, RoomSimulator, AddGaussianNoise, TimeStretch, Normalize
from bengali_speech.tokenizers import dump_default_ood_vocab
from bengali_speech.data_collators import DataCollatorCTCWithPadding
from bengali_speech.evaluate import get_compute_metrics_func
from bengali_speech.kaggle import upload_to_kaggle
from bengali_speech.utils import log_title_with_multiple_lines
from bengali_speech.dataloaders import TrainDataset, ValidationDataset


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
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # report/push
    parser.add_argument("--to_kaggle", action="store_true")
    parser.add_argument("--do_report", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")

    # preprocessing args
    parser.add_argument("--min_sec", type=float, default=2.)
    parser.add_argument("--max_sec", type=float, default=10.)
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
        report_to=["tensorboard", "wandb"] if args.do_report else "none",
        # report_to="none",
        run_name=experiment_name,
    )

    ######################### ALL HUGGINGFACE CONFIG #########################
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
    log_title_with_multiple_lines("Loading DATASETS")
    train_dataset = TrainDataset(tokenizer=tokenizer, feature_extractor=feature_extractor, min_sec=args.min_sec, max_sec=args.max_sec)
    valid_dataset = ValidationDataset(tokenizer=tokenizer, feature_extractor=feature_extractor)

    for dataset in [train_dataset, valid_dataset]:
        logger.info("------")
        logger.info(dataset)
        logger.info(f"{len(dataset)=}")

        example = dataset[0]
        logger.info(f"{example.keys()=}")
        logger.info(f"{len(example['input_values'])=}")
        logger.info(f"{example['labels']=}")


    ######################### LOAD MODEL AND TRAIN #########################
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=get_compute_metrics_func(processor=processor),
        train_dataset=train_dataset,
        eval_dataset={
            "validation": valid_dataset,
            # "example": dataset["example"],
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
        print('KEYBOARD INTERRUPTED!')
        trainer.is_in_train = False

    trainer.save_model()
    trainer.save_state()
    log_title_with_multiple_lines("Done Training and Uploading Output")

    if args.to_kaggle:
        upload_to_kaggle(experiment_name)
        log_title_with_multiple_lines("Done Uploading Output to Kaggle")
