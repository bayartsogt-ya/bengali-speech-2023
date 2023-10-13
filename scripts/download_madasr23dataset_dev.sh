# To download and extract only DEV split
# bash ./scripts/download_madasr23dataset_dev.sh

echo "Downloading Bengali DEV split of madasr23dataset"

OUTPUT_DIR="./data/madasr23dataset"

mkdir -p $OUTPUT_DIR
wget -O "${OUTPUT_DIR}/bn_dev.tar.gz" "https://ee.iisc.ac.in/madasr23dataset/download/bn_dev.tar.gz"

tar -xzf "${OUTPUT_DIR}/bn_dev.tar.gz" -C "${OUTPUT_DIR}"

echo "cloning the label corpus from https://github.com/bloodraven66/RESPIN_ASRU_Challenge_2023"
git clone "https://github.com/bloodraven66/RESPIN_ASRU_Challenge_2023.git" \
    "${OUTPUT_DIR}/RESPIN_ASRU_Challenge_2023"
