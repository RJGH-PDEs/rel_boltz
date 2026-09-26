#!/bin/bash

#SBATCH -N 1
#SBATCH -n 128
#SBATCH -o n5_tensor.%j.out
#SBATCH -e n5_tensor.%j.err
#SBATCH -J boltz_n5
#SBATCH -p normal
#SBATCH -A DMS23021
# 6h: measured chunk 1/17 at 874.8s on this node (idev test, 2026-09-26) ->
# ~4.1h projected total, not the ~1h cross-machine guess. If this still isn't
# enough, run_n5_tensor.py checkpoints after every chunk (results/n5_..._checkpoint.pkl)
# -- just resubmit this same script and it resumes from the last finished chunk.
#SBATCH -t 6:00:00

#SBATCH --mail-user=rodrigogonzalez@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# rel_boltz scripts use relative paths (see CLAUDE.md "Running things"),
# so this must run with cwd = src/, not the repo root.
cd /work/09611/rodrigojosegonzalez/ls6/research/rel_boltz/src

source ~/.bashrc
conda activate ttenv
python3 run_n5_tensor.py
