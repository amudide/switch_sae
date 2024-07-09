python train-relu.py --gpu 6 --l1_penalties 0.001 0.002 0.004
python train-relu.py --gpu 7 --l1_penalties 0.008 0.016 0.032
python train-relu.py --gpu 6 --l1_penalties 0.064 0.128 0.256

python train-anth.py --gpu 0 --l1_penalties 0.03 0.04 0.05
python train-anth.py --gpu 1 --l1_penalties 0.06 0.07 0.08
python train-anth.py --gpu 2 --l1_penalties 0.09 0.10 0.11
python train-anth.py --gpu 3 --l1_penalties 0.12 0.13 0.14