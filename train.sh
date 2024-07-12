python train-moe.py --gpu 0 --ks 8 --num_experts 16 --es 1  --heavisides f t
python train-moe.py --gpu 1 --ks 32 --num_experts 16 --es 1 --heavisides f t
python train-moe.py --gpu 2 --ks 64 --num_experts 16 --es 1 --heavisides f t
python train-moe.py --gpu 3 --ks 128 --num_experts 16 --es 1 --heavisides f t

python train-moe.py --gpu 0 --ks 8 --num_experts 32 --es 1 --heavisides f t
python train-moe.py --gpu 1 --ks 32 --num_experts 32 --es 1 --heavisides f t
python train-moe.py --gpu 2 --ks 64 --num_experts 32 --es 1 --heavisides f t
python train-moe.py --gpu 3 --ks 128 --num_experts 32 --es 1 --heavisides f t