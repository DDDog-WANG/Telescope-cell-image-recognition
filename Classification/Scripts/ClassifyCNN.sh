#!/bin/bash
#$ -cwd
#$ -l rt_G.large=1
#$ -l h_rt=24:00:00

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
module load gcc/11.2.0 python/3.7/3.7.13 cuda/11.0/11.0.3 cudnn/8.1/8.1.1
source ~/python7_env/bin/activate

PYDIR=$HOME/DDDog/Epigenetic/Classification/Scripts
DIR=$HOME/DDDog/Epigenetic/Classification

python $PYDIR/ClassifyCNN_$RESNET.py $DIR $CHIP

