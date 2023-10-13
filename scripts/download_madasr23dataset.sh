# To download and extract FIRST split of total 5 tar.gz dumps
# bash ./scripts/download_madasr23dataset.sh 1
#

echo "Downloading Bengali split of madasr23dataset"

CURRENT=$1
TOTAL=5
OUTPUT_DIR="./data/madasr23dataset"

mkdir -p $OUTPUT_DIR
wget -O "${OUTPUT_DIR}/bn_train_${TOTAL}splits_split${CURRENT}.tar.gz" \
    "https://ee.iisc.ac.in/madasr23dataset/download/bn_train_${TOTAL}splits_split${CURRENT}.tar.gz"

tar -xzf "${OUTPUT_DIR}/bn_train_${TOTAL}splits_split${CURRENT}.tar.gz" -C "${OUTPUT_DIR}"