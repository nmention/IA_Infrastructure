#!/bin/bash
#SBATCH -N1
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -J test_66
#SBATCH -o output.txt
#SBATCH -e error.txt
#SBATCH -p gpu
#SBATCH --gres gpu:rtx6000:1
#SBATCH --time 10-00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=noe.mention@protonmail.com


jupyter nbconvert --to script 'TP4.ipynb'
python3 TP4.py
