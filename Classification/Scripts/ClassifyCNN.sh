#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=10:00:00

# rt_C.small
# 5CPU / 30GB memory / max 168h / 0.2p/h
# rt_C.large
# 20CPU / 120GB memory / max 72h / 0.6p/h
# rt_G.small
# 5CPU / 1GPU / 60GB memory / max 168h / 0.3p/h
# rt_G.large
# 20CPU / 4GPU / 240GB memory / max 72h / 0.9p/h
# See also https://docs.abci.ai/ja/03/
# You can see the list of available modules by "module avail 2>&1 | less"

source /etc/profile.d/modules.sh
module purge
module load gcc/12.2.0 python/3.10/3.10.10 cuda/11.7/11.7.1 cudnn/8.8/8.8.1
source ~/python10_env/bin/activate

PYDIR=$HOME/DDDog/Epigenetic/Classification/Scripts
DIR=$HOME/DDDog/Epigenetic/Classification

model_dir=$DIR/results/2023/Train/${RESNET}_${CHIP}
result_dir=$DIR/Models/${RESNET}_${CHIP}
if [ ! -d "$model_dir" ]; then
    mkdir $model_dir
    echo "mkdir $model_dir"
fi
if [ ! -d "$result_dir" ]; then    
    mkdir $result_dir
    echo "mkdir $result_dir"
fi
python $PYDIR/ClassifyCNN.py $DIR $CHIP $RESNET

