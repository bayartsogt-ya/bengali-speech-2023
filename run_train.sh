# Run train.py with the given arguments
python train.py \
    --base_model_name "" \
    --experiment_number "1" \
    --wav2vec_freeze_feature_extractor \
    --batch_size 8 \
    --num_train_epochs 1 \
    --dataloader_num_workers 16 \
    # --num_proc 8 \
    # --fp16 \
    # --to_kaggle \
    # --push_to_hub \
    
    # --min_sec "5"\
    # --max_sec "10" \
    # --train_percentage "0.3" \

