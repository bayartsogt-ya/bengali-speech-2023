
OUTPUT_DIR="./data/bengaliai-speech"

kaggle competitions download -c bengaliai-speech -p ${OUTPUT_DIR}
unzip ${OUTPUT_DIR}/bengaliai-speech.zip -d ${OUTPUT_DIR}
kaggle datasets download -f annoated.csv -p ${OUTPUT_DIR} mbmmurad/ood-example-audios-hand-annotations
