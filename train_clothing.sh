for i in $(seq 3 3)
    do
        echo 0.$i
        python open_lth.py lottery \
        --default_hparams=clothing_resnet_50\
        --levels=3\
        --lr=0.02\
        --batch_size=32\
        --pruning_fraction 0.$i \
        --replicate=5 \
        --dataset='clothing' \
        --e1=20
    done