python train-relu.py --gpu 0 --l1_penalties 5 6.46
python train-relu.py --gpu 1 --l1_penalties 8.34 10.78
python train-relu.py --gpu 2 --l1_penalties 13.92 17.98
python train-relu.py --gpu 3 --l1_penalties 23.23 30

python train-gated.py --gpu 0 --l1_penalties 10 13.20
python train-gated.py --gpu 6 --l1_penalties 17.43 23.01
python train-gated.py --gpu 6 --l1_penalties 30.39 40.14
python train-gated.py --gpu 7 --l1_penalties 53.01 60

python train-topk.py --gpu 4 --ks 8 16
python train-topk.py --gpu 5 --ks 32 48
python train-topk.py --gpu 6 --ks 64 96
python train-topk.py --gpu 7 --ks 128 192

cd /data/cb/scratch/amudide/switch_sae
conda activate sae