python train-switch.py --gpu 0 --ks 8 16 --num_experts 16
python train-switch.py --gpu 1 --ks 32 48 --num_experts 16
python train-switch.py --gpu 2 --ks 64 96 --num_experts 16
python train-switch.py --gpu 3 --ks 128 192 --num_experts 16

python train-switch.py --gpu 3 --ks 8 16 --num_experts 32
python train-switch.py --gpu 4 --ks 32 48 --num_experts 32
python train-switch.py --gpu 5 --ks 64 96 --num_experts 32
python train-switch.py --gpu 6 --ks 128 192 --num_experts 32

python train-switch.py --gpu 4 --ks 8 16 --num_experts 64
python train-switch.py --gpu 5 --ks 32 48 --num_experts 64
python train-switch.py --gpu 6 --ks 64 96 --num_experts 64
python train-switch.py --gpu 7 --ks 128 192 --num_experts 64

python train-switch.py --gpu 6 --ks 8 16 32 48 --num_experts 128
python train-switch.py --gpu 7 --ks 64 96 128 192 --num_experts 128

python train-switch-flop.py --gpu 3 --ks 8 16 32 --num_experts 4
python train-switch-flop.py --gpu 4 --ks 64 128 192 --num_experts 4

python train-switch-flop.py --gpu 6 --ks 8 16 32 --num_experts 8
python train-switch-flop.py --gpu 7 --ks 64 128 192 --num_experts 8

## TODO

python train-switch-flop.py --gpu 3 --ks 64 --num_experts 1 2
python train-switch-flop.py --gpu 4 --ks 64 --num_experts 4
python train-switch-flop.py --gpu 6 --ks 64 --num_experts 8
python train-switch-flop.py --gpu 7 --ks 64 --num_experts 16

