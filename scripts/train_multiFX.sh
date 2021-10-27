python src/train_multiFX.py \
        dataset/generated/gen_multiFX_guitarset_[1,5]/ \
        --max_epoch 10 \
        --gpus 1 \
        --val_check_interval 0.5 \
        --random_seed 42 \
        --log_class_loss true\
