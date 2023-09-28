

def get_prepare_dataset_func(tokenizer, feature_extractor):
    def prepare_dataset(batch):
        batch["input_values"] = feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    return prepare_dataset

def get_filter_by_length_func(min_sec: float, max_sec: float):
    def filter_by_length(batch):
        duration = batch["audio"]["array"].shape[0] / batch["audio"]["sampling_rate"]
        return min_sec < duration < max_sec
    return filter_by_length
