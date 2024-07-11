python train-moe.py --gpu 0 --ks 8 16 --num_experts 16 --es 1
python train-moe.py --gpu 1 --ks 32 48 --num_experts 16 --es 1
python train-moe.py --gpu 2 --ks 64 96 --num_experts 16 --es 1
python train-moe.py --gpu 3 --ks 128 192 --num_experts 16 --es 1

python train-moe.py --gpu 0 --ks 8 16 --num_experts 32 --es 1
python train-moe.py --gpu 1 --ks 32 48 --num_experts 32 --es 1
python train-moe.py --gpu 2 --ks 64 96 --num_experts 32 --es 1
python train-moe.py --gpu 3 --ks 128 192 --num_experts 32 --es 1