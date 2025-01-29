#!/bin/bash

#SBATCH --job-name='test_gen'
#SBATCH --nodes=1   
#SBATCH --ntasks=16                 
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH -p general              
#SBATCH -q debug
            
#SBATCH -t 00-00:15:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate edm

# echo $(which python)
dataset_name=brats21
data_dir=/data/amciilab/yiming/DATA/BraTS21_training/preprocessed_data_flair_00_128/np_val
model=/data/amciilab/yiming/projects/Diffusion/edm/training-runs/00011-brats21-uncond-ddpmpp-vp-gpus1-batch128-fp32/network-snapshot-002224.pkl

data="--data=$data_dir --dataset_name=$dataset_name"

S_churn=10
S_min=1.0
S_max=10.0
S_noise=1.003
hyperparameterS="--S_churn=$S_churn --S_min=$S_min --S_max=$S_max --S_noise=$S_noise"

Ablate="--solver=heun --disc=vp --schedule=vp --scaling=vp"

Recon="--steps-noise=100 --steps-denoise=100"
Postrecon="--threshold=0.1 --mixed=True --cc-filter=False"

Custom="--gaussian=True --grad=False"

torchrun --standalone --nproc_per_node=1 detect.py --outdir=detecting-runs --seed=0 --batch_size=2 --total_kimg=2 --network=$model $hyperparameterS $Ablate $Postrecon $Custom $data $Recon
