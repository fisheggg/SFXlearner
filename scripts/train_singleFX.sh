python src/train_singleFX.py \
        dataset/generated/gen_singleFX_1onN_09182021/ \
        --max_epoch 100 \
        --gpus 1 \
        --check_val_every_n_epoch 1 \
        --with_clean 1