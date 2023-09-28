import logging
import evaluate
import numpy as np


logger = logging.getLogger(__name__)


def get_compute_metrics_func(processor):
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = metric_wer.compute(predictions=pred_str, references=label_str)
        cer = metric_cer.compute(predictions=pred_str, references=label_str)

        return {
            "wer": wer,
            "cer": cer,
        }
    
    return compute_metrics
