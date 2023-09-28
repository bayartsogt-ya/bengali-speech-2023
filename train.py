"""Entry point for training the model."""

import os
import re
import argparse
import logging
from bengali_speech.datasets import read_bengaliai_speech_2023
from bengali_speech.tokenizers import get_default_wav2vec_tokenizer
from bengali_speech.data_collators import DataCollatorCTCWithPadding
from bengali_speech.evaluate import get_compute_metrics_func
from bengali_speech.kaggle import upload_to_kaggle
from bengali_speech.utils import log_title_with_multiple_lines


from transformers import AutoFeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments



# ignoring mac error

# set up logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger.info(f"PYTORCH_ENABLE_MPS_FALLBACK = {os.getenv('PYTORCH_ENABLE_MPS_FALLBACK')}")

    # parameters
    base_model_name = "patrickvonplaten/tiny-wav2vec2-no-tokenizer"
    tokenizer_name = None

    experiment_number = 0

    wav2vec_freeze_feature_extractor = True

    batch_size = 1
    num_train_epochs = 1
    dataloader_num_workers = 1

    to_kaggle = False
    push_to_hub = False

    experiment_name = f"bengali-2023-{experiment_number:04d}"

    training_args = TrainingArguments(
        output_dir=os.path.join("output", experiment_name),
        group_by_length=True,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # num_train_epochs=num_train_epochs,
        max_steps=2,
        dataloader_num_workers=dataloader_num_workers,
        
        # LR
        learning_rate=3e-4,
        weight_decay=0.005,
        warmup_ratio=0.1,
        # warmup_steps=500,

        # EVAL & SAVE
        logging_steps=10,
        
        evaluation_strategy="steps",
        eval_steps=2,  # eval 20 times

        save_strategy="steps",
        save_steps=2,  # save 20 times
        save_total_limit=3,
        
        fp16=False, # -------------------------------------> fp16

        load_best_model_at_end=True,
        log_level="debug",

        # report
        push_to_hub=push_to_hub,
        metric_for_best_model="validation_wer",
        # report_to=["tensorboard", "wandb"],
        report_to="none",
        run_name=experiment_name,
    )

    # read bengali speech 2023 competition data
    log_title_with_multiple_lines("Reading data, tokenizer, and feature extractor.")
    dataset = read_bengaliai_speech_2023(path_to_data="data/bengaliai-speech")
    logger.info(dataset)

    # load tokenizer
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = get_default_wav2vec_tokenizer()

    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    keep_chars = "".join(tokenizer.vocab)
    logger.critical("Keep only following characters: %s", keep_chars)

    def clean_text(text: str):
        return re.sub(f"[^{keep_chars}]", "", text)

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = tokenizer(clean_text(batch["sentence"])).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

    logger.info("Done preparing dataset.")

    # data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    log_title_with_multiple_lines("Loading Model and Start Training.")
    # load model
    model = Wav2Vec2ForCTC.from_pretrained(
        base_model_name,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )

    if hasattr(model, "freeze_feature_extractor") and wav2vec_freeze_feature_extractor:
        logger.info("Freezing a feature extractor.")
        model.freeze_feature_extractor()

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=get_compute_metrics_func(processor=processor),
        train_dataset=dataset["train"],
        eval_dataset={
            "validation": dataset["validation"],
            # "example": dataset["example"][:2],
        },
        tokenizer=processor.feature_extractor,
    )

    train_result = trainer.train()

    log_title_with_multiple_lines("Done Training and Uploading Output")

    tokenizer.save_pretrained(training_args.output_dir)
    trainer.model.save_pretrained(training_args.output_dir)
    feature_extractor.save_pretrained(training_args.output_dir)

    if to_kaggle:
        upload_to_kaggle(experiment_name)

    # create a kaggle dataset from trainer output.
    # https://www.kaggle.com/docs/api
    # kaggle datasets create -p bengali-2023-000
