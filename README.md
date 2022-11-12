## When Sparse Neural Network Meets Label Noise Learning: A Multi-Stage Learning Framework

### Welcome

### Notation

Codes for our paper ``When Sparse Neural Network Meets Label Noise Learning: A Multi-Stage Learning Framework`` (TNNLS 2022).
Ubuntu 16 or 18 are ok for adoption.

### Run
If you wish to run this method, you can conduct with the following shell comments:

CIFAR-10：
``
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
``

CIFAR-100：
``
for i in $(seq 3 6)
    do
        echo 0.$i
        python open_lth.py lottery \
        --default_hparams=cifar_vgg_16\
        --levels=3\
        --batch_size=256\
        --pruning_fraction 0.$i \
        --replicate=459\
        --dataset='cifar100' \
        --noise_ratio=0.2 \
        --noise_type='sym' \
        --training_steps='120ep'\
        --e1=20 \
        --e2=80
    done
``

Before running the comments, please remember to change the dataset root in foundations/hparams.py.
### Acknowledgements
Thanks for the authors of LTH which offer such a well-organized framework.  
