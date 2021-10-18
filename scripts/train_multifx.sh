python src/train_multiFX.py \
        dataset/generated/gen_multiFX_[1,5]/ \
        --max_epoch 10 \
        --gpus 1 \
        --log_class_loss true \
        --val_check_interval 500 \
        --with_clean false