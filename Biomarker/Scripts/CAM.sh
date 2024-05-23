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

HOMEPATH=$HOME/DDDog/Epigenetic/Classification
PYDIR=$HOMEPATH/Scripts/CAM.py
STAINTYPE=$STAINTYPE
MODELTYPE=Resnet10_noavg
CTRLTYPE=$CTRLTYPE
IMAGEPATH=$HOMEPATH/Datasets
MODELPATH=$HOMEPATH/Models
CAMTYPE=ScoreCAM
CAMPATH=$HOMEPATH/results_cam/${CTRLTYPE}_${STAINTYPE}_${MODELTYPE}_${CAMTYPE}
TARGET_LAYER="model.resnet.layer2"
if [ ! -d "$CAMPATH" ]; then
    mkdir $CAMPATH
    echo "mkdir $CAMPATH"
fi
echo "#############################"
echo 👑 BASH SCRIPT START
echo PYDIR: $PYDIR
echo STAINTYPE: "$STAINTYPE"
echo MODELTYPE: "$MODELTYPE"
echo CTRLTYPE: "$CTRLTYPE"
echo IMAGEPATH: "$IMAGEPATH"
echo MODELPATH: "$MODELPATH"
echo CAMPATH: "$CAMPATH"
echo CAMTYPE: "$CAMTYPE"
echo TARGET_LAYER: "$TARGET_LAYER"
echo "#############################"
python $PYDIR --stain_type $STAINTYPE --ctrl_type $CTRLTYPE --model_type $MODELTYPE --image_path $IMAGEPATH --model_path $MODELPATH --cam_path $CAMPATH --cam_type $CAMTYPE --target_layer $TARGET_LAYER

