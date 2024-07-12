python train-moe.py --gpu 5 --ks 8 --num_experts 16 --es 1  --heavisides f t
python train-moe.py --gpu 6 --ks 32 --num_experts 16 --es 1 --heavisides f t
python train-moe.py --gpu 7 --ks 64 --num_experts 16 --es 1 --heavisides f t
python train-moe.py --gpu 6 --ks 128 --num_experts 16 --es 1 --heavisides f t

python train-moe.py --gpu 5 --ks 8 --num_experts 32 --es 1 --heavisides f t
python train-moe.py --gpu 5 --ks 32 --num_experts 32 --es 1 --heavisides f t
python train-moe.py --gpu 6 --ks 64 --num_experts 32 --es 1 --heavisides f t
python train-moe.py --gpu 7 --ks 128 --num_experts 32 --es 1 --heavisides f t