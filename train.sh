## ADD WANDB NAME TO ARGPARSE FOR SH FILES!

python train-switch.py --gpu 5 --ks 8 --num_experts 16 --heavisides f t
python train-switch.py --gpu 6 --ks 32 --num_experts 16 --heavisides f t
python train-switch.py --gpu 7 --ks 64 --num_experts 16 --heavisides f t
python train-switch.py --gpu 4 --ks 128 --num_experts 16 --heavisides f t

python train-switch.py --gpu 5 --ks 8 --num_experts 32 --heavisides f t
python train-switch.py --gpu 5 --ks 32 --num_experts 32 --heavisides f t
python train-switch.py --gpu 6 --ks 64 --num_experts 32 --heavisides f t
python train-switch.py --gpu 7 --ks 128 --num_experts 32 --heavisides f t

python train-switch-flop.py --gpu 2 --ks 8 32 --num_experts 8 --heavisides f
python train-switch-flop.py --gpu 3 --ks 64 128 --num_experts 8 --heavisides f

## Test LB

python train-switch.py --gpu 5 --ks 64 --num_experts 16 --lb_alphas 0 0.001
python train-switch.py --gpu 6 --ks 64 --num_experts 16 --lb_alphas 0.003 0.01
python train-switch.py --gpu 7 --ks 64 --num_experts 16 --lb_alphas 0.03 0.1

python train-switch.py --gpu 5 --ks 64 --num_experts 16 --lb_alphas 0.3 1
python train-switch.py --gpu 6 --ks 64 --num_experts 16 --lb_alphas 3 10
python train-switch.py --gpu 7 --ks 64 --num_experts 16 --lb_alphas 30 100


cd /data/cb/scratch/amudide/switch_sae
conda activate sae