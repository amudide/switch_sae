python train-relu.py --gpu 0 --l1_penalties 5 6.46
python train-relu.py --gpu 1 --l1_penalties 8.34 10.78
python train-relu.py --gpu 2 --l1_penalties 13.92 17.98
python train-relu.py --gpu 3 --l1_penalties 23.23 30

python train-gated.py --gpu 4 --l1_penalties 0.01 0.03
python train-gated.py --gpu 5 --l1_penalties 0.09 0.27
python train-gated.py --gpu 6 --l1_penalties 0.81 2.43
python train-gated.py --gpu 7 --l1_penalties 7.29 15

python train-topk.py --gpu 4 --ks 8 16
python train-topk.py --gpu 5 --ks 32 48
python train-topk.py --gpu 6 --ks 64 96
python train-topk.py --gpu 7 --ks 128 192

cd /data/cb/scratch/amudide/switch_sae
conda activate sae