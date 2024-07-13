## ADD WANDB NAME TO ARGPARSE FOR SH FILES!

python train-switch.py --gpu 5 --ks 8 --num_experts 16 --heavisides f t
python train-switch.py --gpu 6 --ks 32 --num_experts 16 --heavisides f t
python train-switch.py --gpu 7 --ks 64 --num_experts 16 --heavisides f t
python train-switch.py --gpu 4 --ks 128 --num_experts 16 --heavisides f t

python train-switch.py --gpu 5 --ks 8 --num_experts 32 --heavisides f t
python train-switch.py --gpu 5 --ks 32 --num_experts 32 --heavisides f t
python train-switch.py --gpu 6 --ks 64 --num_experts 32 --heavisides f t
python train-switch.py --gpu 7 --ks 128 --num_experts 32 --heavisides f t

cd /data/cb/scratch/amudide/switch_sae
conda activate sae