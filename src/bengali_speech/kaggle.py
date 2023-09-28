import subprocess
import logging
import json


logger = logging.getLogger(__name__)

def save_dataset_metadata(dataset_name):
    metadata = {}
    metadata["title"] = dataset_name
    metadata["id"] = f"bayartsogtya/{dataset_name}"
    metadata["licenses"] = [{"name": "CC0-1.0"}]
    with open(f"output/{dataset_name}/dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Created dataset-metadata.json")


def upload_to_kaggle(dataset_name):
    save_dataset_metadata(dataset_name)
    command = f"kaggle datasets create -p output/{dataset_name}"
    logger.info("Run the following command to upload to kaggle:")
    logger.info(command)
    subprocess.check_call(command.split(" "))
    logger.info("Done uploading to kaggle.")
