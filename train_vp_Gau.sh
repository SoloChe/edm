#!/bin/bash
#SBATCH --job-name='test_train'
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1               
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH -p general                
#SBATCH -q debug
            
#SBATCH -t 00-00:10:00               
            
#SBATCH -e ./slurm_out/slurm.%j.err
#SBATCH -o ./slurm_out/slurm.%j.out


module purge
module load mamba/latest
source activate edm

dataset_name=brats21
data_dir=/data/amciilab/yiming/DATA/BraTS21_training/preprocessed_data_flair_00_128/np_train_healthy

# slurm setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_PORT

# resume=/data/amciilab/yiming/projects/Diffusion/edm/training-runs/00005-brats21_train_all_healthy-uncond-ddpmpp-vp-gpus2-batch128-fp32/training-state-005094.pt


torchrun --nproc_per_node=1\
         --nnodes=1\
         --rdzv_backend=c10d\
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\
         train.py --outdir=training-runs --data=$data_dir --cond=0 --arch=ddpmpp --batch-gpu=32 --tick=10 --dump=10 --snap=10 --workers=4 --batch=128 --cache=False --augment=0 --arch=ddpmpp --precond=vp --lr=2e-4 --ema=0.9 --dropout=0.10 --gaussian=True --grad=False --dataset_name=$dataset_name


