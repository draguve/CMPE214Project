#!/bin/bash
#SBATCH -o ./%j.%x.out
#SBATCH -e ./%j.%x.err
#SBATCH --job-name=ddp-cmpe214    # create a short name for your job
#SBATCH --partition=gpuqm
#SBATCH --time=3-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --export=ALL
#SBATCH --nodes=4
#SBATCH --nodelist=g1,g2,g3,g4,g5,g7,g8,g9,g10,g11,g12,g13,g14,g15
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

export http_proxy=
export https_proxy=
export ALL_PROXY=
export MASTER_PORT=16000
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "Nodes assigned: $SLURM_NODELIST"
echo "Running on $(scontrol show job $SLURM_JOB_ID | grep NumNodes)"

srun python test.py --num-sequences 1000000 --batch-size 32 --no-tqdm-enabled --backend gloo --results-file final_results.csv
srun python test.py --num-sequences 1000000 --batch-size 64 --no-tqdm-enabled --backend gloo --results-file final_results.csv
srun python test.py --num-sequences 1000000 --batch-size 128 --no-tqdm-enabled --backend gloo --results-file final_results.csv
srun python test.py --num-sequences 1000000 --batch-size 256 --no-tqdm-enabled --backend gloo --results-file final_results.csv
