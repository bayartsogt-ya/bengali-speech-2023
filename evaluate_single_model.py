import time
import logging
import argparse
from tqdm.auto import tqdm

import pandas as pd

import evaluate
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from transformers.pipelines.pt_utils import KeyDataset
from pyctcdecode import BeamSearchDecoderCTC

from bengali_speech.datasets.bengaliai_speech import read_bengaliai_speech_2023_using_hf_datasets
from bengali_speech.utils import log_title_with_multiple_lines


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_title_with_multiple_lines("Starting Evaluation:")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="./output/wav2vec2-xls-r-300m-bengali")
    parser.add_argument("--kenlm_model", type=str, default="./output/wav2vec2-xls-r-300m-bengali")
    args = parser.parse_args()

    # print args nicely
    for arg in vars(args):
        logger.info("%s: %s", arg, getattr(args, arg))

    # load pipeline
    st = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model_name)
    decoder = BeamSearchDecoderCTC.load_from_dir(args.kenlm_model)
    logger.info(f"Done loading tokenizer, feature extractor, kenlm decoder in {time.time() - st:.2f} seconds.")

    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    logger.info(decoder)
    
    # load model
    model = AutoModelForCTC.from_pretrained(args.base_model_name)

    logger.info("Done loading config, tokenizer, feature extractor, and model.")
    dataset_dict = read_bengaliai_speech_2023_using_hf_datasets("data/bengaliai-speech")

    result_dict = []

    for _decoder in [None, decoder]:
        decoder_kwargs = {"decoder": _decoder}
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            **decoder_kwargs,
        )

        for _split in ["validation", "example"]:
            logger.info(f"{_decoder=} {_split=}")
            
            st = time.time()
            current_ds = dataset_dict[_split]
            references = dataset_dict[_split]["sentence"]
            predictions = [out["text"] for out in tqdm(pipe(KeyDataset(dataset_dict[_split], "audio"), batch_size=2))]

            _wer = metric_wer.compute(predictions=predictions, references=references)
            _cer = metric_cer.compute(predictions=predictions, references=references)

            result_dict.append({
                "split": _split,
                "kenlm": _decoder is not None,
                "wer": _wer,
                "cer": _cer,
            })

            logger.info(f"Done inferencing in {time.time() - st:.2f} seconds.")
    
    logger.info("\n"*2 + pd.DataFrame(result_dict).to_markdown())
