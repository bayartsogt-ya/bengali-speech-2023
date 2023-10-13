
OUTPUT_DIR="./data/bengaliai-speech"

kaggle competitions download -c bengaliai-speech -p ${OUTPUT_DIR}
unzip ${OUTPUT_DIR}/bengaliai-speech.zip -d ${OUTPUT_DIR}

kaggle datasets download -f annoated.csv -p ${OUTPUT_DIR} mbmmurad/ood-example-audios-hand-annotations

kaggle datasets download -f train_metadata.csv -p ${OUTPUT_DIR} imtiazprio/bengaliai-speech-train-metadata
unzip ${OUTPUT_DIR}/train_metadata.csv -d ${OUTPUT_DIR}

kaggle datasets download -f NISQA_wavfiles.csv -p ${OUTPUT_DIR} imtiazprio/bengaliai-speech-train-nisqa
unzip ${OUTPUT_DIR}/NISQA_wavfiles.csv -d ${OUTPUT_DIR}
