"""Entry point for training the model."""

import os
import re
import argparse
import logging
from bengali_speech.datasets import read_bengaliai_speech_2023_using_hf_datasets
from bengali_speech.mappers import get_filter_by_length_func, get_prepare_dataset_func
from bengali_speech.tokenizers import get_default_wav2vec_tokenizer
from bengali_speech.data_collators import DataCollatorCTCWithPadding
from bengali_speech.evaluate import get_compute_metrics_func
from bengali_speech.kaggle import upload_to_kaggle
from bengali_speech.utils import log_title_with_multiple_lines


from transformers import AutoFeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments


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
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=12)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--to_kaggle", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--fp16", action="store_true")

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

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.num_train_epochs,
        dataloader_num_workers=args.dataloader_num_workers,
        
        # LR
        learning_rate=3e-4,
        weight_decay=0.005,
        warmup_ratio=0.1,
        # warmup_steps=500,

        # EVAL & SAVE
        logging_steps=100,
        
        evaluation_strategy="steps",
        eval_steps=500,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        
        fp16=args.fp16,

        load_best_model_at_end=True,
        log_level="debug",

        # report
        push_to_hub=args.push_to_hub,
        metric_for_best_model="eval_validation_wer",
        report_to=["tensorboard", "wandb"],
        # report_to="none",
        run_name=experiment_name,
    )

    ######################### DATA PREPROCESSING #########################
    # read bengali speech 2023 competition data
    log_title_with_multiple_lines("Reading data, tokenizer, and feature extractor.")
    dataset = read_bengaliai_speech_2023_using_hf_datasets(path_to_data="data/bengaliai-speech")
    if args.num_shards_train:
        dataset["train"] = dataset["train"].shard(num_shards=args.num_shards_train, index=args.shard_index_train)
    if args.num_shards_validation:
        dataset["validation"] = dataset["validation"].shard(num_shards=args.num_shards_validation, index=args.shard_index_validation)
    logger.info(dataset)

    # load tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = get_default_wav2vec_tokenizer()

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model_name)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    keep_chars = "".join(tokenizer.vocab)
    logger.critical("Keep only following characters: %s", keep_chars)

    def clean_text(batch):
        batch["sentence"] = re.sub(f"[^{keep_chars}]", "", batch["sentence"])
        return batch

    dataset = dataset.map(clean_text, num_proc=args.num_proc)
    logger.info("After cleaning:")
    logger.info(dataset)

    def filter_by_length(batch):
        duration = batch["audio"]["array"].shape[0] / batch["audio"]["sampling_rate"]
        return args.min_sec < duration < args.max_sec

    if args.min_sec and args.max_sec:
        dataset["train"] = dataset["train"].filter(filter_by_length, num_proc=args.num_proc)
        logger.info("After filtering:")
        logger.info(dataset)
    else:
        logger.info("Skip filtering... (min_sec and max_sec are not set)")

    def prepare_dataset(batch):
        batch["input_values"] = feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=args.num_proc, writer_batch_size=1000)

    logger.info("Done preparing dataset.")

    ######################### LOAD MODEL AND TRAIN #########################
    # data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    log_title_with_multiple_lines("Loading Model and Start Training.")
    # load model
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model_name,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )

    if hasattr(model, "freeze_feature_encoder") and args.wav2vec_freeze_feature_extractor:
        logger.info("Freezing a feature extractor.")
        model.freeze_feature_encoder()

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

    logger.info("Start training...")
    train_result = trainer.train()

    log_title_with_multiple_lines("Done Training and Uploading Output")

    tokenizer.save_pretrained(training_args.output_dir)
    trainer.model.save_pretrained(training_args.output_dir)
    feature_extractor.save_pretrained(training_args.output_dir)

    if args.to_kaggle:
        upload_to_kaggle(experiment_name)
