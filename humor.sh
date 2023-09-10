#!/bin/bash
dir=$(pwd)
echo "Current root $dir"
export PATH=/vol/bitbucket/yg2719/miniconda3/bin:/vol/bitbucket/yg2719/miniconda3/envs/humor_env/bin/:$PATH
source /vol/bitbucket/yg2719/miniconda3/etc/profile.d/conda.sh
cd /vol/bitbucket/yg2719/IndividualProject/humor
conda activate humor_env
python myProject/main.py $1 #@./configs/test_humor_sampling.cfg