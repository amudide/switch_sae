## ADD WANDB NAME TO ARGPARSE FOR SH FILES!

python train-topk.py --gpu 3 --ks 8 16 32 48 --dict_ratio 64
python train-topk.py --gpu 4 --ks 64 96 128 192 --dict_ratio 64

python train-topk.py --gpu 4 --ks 8 16
python train-topk.py --gpu 5 --ks 32 48
python train-topk.py --gpu 6 --ks 64 96
python train-topk.py --gpu 7 --ks 128 192

python train-relu.py --gpu 4 --l1_penalties 5 6.46
python train-relu.py --gpu 5 --l1_penalties 8.34 10.78
python train-relu.py --gpu 6 --l1_penalties 13.92 17.98
python train-relu.py --gpu 7 --l1_penalties 23.23 30
python train-relu.py --gpu 5 --l1_penalties 38.71 50
python train-relu.py --gpu 0 --l1_penalties 56 64
python train-relu.py --gpu 1 --l1_penalties 75 95
python train-relu.py --gpu 2 --l1_penalties 1 2
python train-relu.py --gpu 3 --l1_penalties 3 4

python train-gated.py --gpu 3 --l1_penalties 1 1.46
python train-gated.py --gpu 4 --l1_penalties 2.13 3.11
python train-gated.py --gpu 5 --l1_penalties 4.53 6.62
python train-gated.py --gpu 6 --l1_penalties 9.65 14.09
python train-gated.py --gpu 7 --l1_penalties 20.56 30
python train-gated.py --gpu 4 --l1_penalties 5 6


cd /data/cb/scratch/amudide/switch_sae
conda activate sae