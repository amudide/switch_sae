python train-relu.py --gpu 0 --l1_penalties 0.03 0.05
python train-relu.py --gpu 1 --l1_penalties 0.07 0.09
python train-relu.py --gpu 2 --l1_penalties 0.11 0.13

python train-gated.py --gpu 4 --l1_penalties 0.03 0.05
python train-gated.py --gpu 5 --l1_penalties 0.07 0.09
python train-gated.py --gpu 6 --l1_penalties 0.11 0.13

python train-topk.py --gpu 0 --ks 8 16
python train-topk.py --gpu 1 --ks 32 64
python train-topk.py --gpu 2 --ks 128 192
