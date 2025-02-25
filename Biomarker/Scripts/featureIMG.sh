#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=1:00:00

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
module load gcc/13.2.0 python/3.10/3.10.14 cuda/11.8/11.8.0 cudnn/8.8/8.8.1
source ~/python10_env/bin/activate

HOMEPATH=$HOME/DDDog/Epigenetic
PYDIR=$HOMEPATH/Biomarker/featureIMG.py
CTRLTYPE=$CTRLTYPE
IMAGEPATH=$HOMEPATH/Classification/Datasets
SAVEPATH=$HOMEPATH/Biomarker

echo "#############################"
echo 👑 BASH SCRIPT START
echo PYDIR: $PYDIR
echo CTRLTYPE: "$CTRLTYPE"
echo IMAGEPATH: "$IMAGEPATH"
echo SAVEPATH: "$SAVEPATH"
echo "#############################"
python $PYDIR --ctrl_type $CTRLTYPE --image_path $IMAGEPATH --save_path $SAVEPATH

