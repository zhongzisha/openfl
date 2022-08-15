#!/bin/bash

#SBATCH --job-name debug_openfl
#SBATCH --partition=gpu
#SBATCH --mail-user=ziszhong2022.slurm@gmail.com
#SBATCH --mail-type=ALL
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:p100:1,lscratch:512
#SBATCH --constraint=gpup100
#SBATCH --cpus-per-task=16
#SBATCH --mem=90g
#SBATCH --ntasks-per-core=1
#SBATCH --time=240:00:00




sleep 100000000000000000000

#800 cases
#4parts, 4nodes
#1000  -> 250 per slide
#5000  -> 1250 per slide
#10000 -> 2500 per slide

