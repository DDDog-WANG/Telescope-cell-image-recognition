#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=6:00:00

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

python $PYDIR/profilecam.py $CHIP $CAM $DIR $DRAG

