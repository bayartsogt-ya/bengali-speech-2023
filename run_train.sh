# Run train.py with the given arguments
python train.py \
    --base_model_name "facebook/wav2vec2-xls-r-300m" \
    --experiment_number "14" \
    --wav2vec_freeze_feature_extractor \
    --max_steps "480000" \
    --batch_size "4" \
    --gradient_accumulation_steps "2" \
    --dataloader_num_workers "16" \
    --num_proc "16" \
    --min_sec "0"\
    --max_sec "15" \
    --num_shards_validation "20" \
    --resume_from_checkpoint "output/bengali-2023-0014/checkpoint-68000" \
    --fp16 \
    --to_kaggle
    # --num_train_epochs "10" \
    # --num_shards_train "3" \
    # --push_to_hub \
