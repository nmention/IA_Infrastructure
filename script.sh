#!/bin/bash
#SBATCH -N1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH -J test_66
#SBATCH -o output.txt
#SBATCH -e error.txt
#SBATCH --time 10-00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=noe.mention@protonmail.com


jupyter nbconvert --to script 'TP4_bis.ipynb'
python3 TP4_bis.py
