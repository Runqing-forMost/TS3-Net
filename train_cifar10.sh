for i in $(seq 3 6)
    do
        echo 0.$i
        CUDA_VISIBLE_DEVICES = 1
        python open_lth.py lottery \
        --default_hparams=cifar_cnn\
        --levels=3\
        --batch_size=256\
        --pruning_fraction 0.$i \
        --replicate=1217\
        --dataset='cifar10' \
        --noise_ratio=0.4 \
        --noise_type='sym' \
        --training_steps='120ep' \
        --e1=20 \
        --e2=80
    done