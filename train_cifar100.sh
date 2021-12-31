for i in $(seq 4 4)
    do
        echo 0.$i
        python open_lth.py lottery \
        --default_hparams=cifar_vgg_16\
        --levels=3\
        --batch_size=256\
        --pruning_fraction 0.$i \
        --replicate=28\
        --dataset='cifar100' \
        --noise_ratio=0.2 \
        --noise_type='sym' \
        --training_steps='120ep' \
        --e1=20\
        --e2=80
    done