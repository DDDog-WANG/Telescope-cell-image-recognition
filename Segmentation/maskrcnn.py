# 0. Import Libraries and Mount Drive
print("0. Import Libraries and Mount Drive", flush=True)
import warnings
import logging
warnings.filterwarnings("ignore")
import os,cv2,sys
from skimage import io
from tqdm import tqdm
from scipy import stats
import shutil
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.measure import label, regionprops
import tensorflow as tf
import keras
tf.get_logger().setLevel(logging.ERROR)

from imgaug import augmenters as iaa
from datetime import timedelta
import datetime
import imageio
import math
import pytz
from pytz import timezone
import imagecodecs._imcd
from tifffile import imread

from root.Mask_RCNN.mrcnn.config import Config
from root.Mask_RCNN.mrcnn import utils
from root.Mask_RCNN.mrcnn import model as modellib
from root.Mask_RCNN.mrcnn import visualize
from root.Mask_RCNN.mrcnn import postprocessing
from root.Mask_RCNN.mrcnn.model import log
import nucleus
print("##########################################################", flush=True)

imagepath=sys.argv[1]
imagename=sys.argv[2]
savepath=sys.argv[3]
weightpath=sys.argv[4]
print("imagepath is ",imagepath, flush=True)
print("imagename is ",imagename, flush=True)
print("savepath is ",savepath, flush=True)

def gamma_img(gamma, img):
    gamma_cvt = np.zeros((256,1), dtype=np.uint8)
    for i in range(256):
        gamma_cvt[i][0] = 255*(float(i)/255)**(1.0/gamma)
    return cv2.LUT(img, gamma_cvt)

# 1. Load and Process Images
print("1. Load and Process Images", flush=True)
img_origin = io.imread(path)
img_uint8 = img_origin - img_origin.min()
img_uint8 = img_uint8 / (img_uint8.max() - img_uint8.min())
img_uint8 *= 255
img_uint8 = img_uint8.astype(np.uint8)

b = img_uint8[:,:,2]
img_hoechst = cv2.merge([b,b,b])
img_hoechst = gamma_img(0.7, img_hoechst)
print("##########################################################", flush=True)

# 2. Configuration
print("2. Configuration", flush=True)
class NucleusInferenceConfig(nucleus.NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MAX_DIM = 512

    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.5
    RPN_NMS_THRESHOLD = 0.3
    
    BACKBONE = "resnet50"
    
    MEAN_PIXEL = np.mean(img_hoechst,axis=(0,1))
    
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (100, 100)
print("##########################################################", flush=True)
        
# 3. Run the Network
print("3. Run the Network", flush=True)
Weights = "Kaggle"
if Weights == "Kaggle":
    weights_path = weightpath+'/mask_rcnn_kaggle_v1.h5' 
elif Weights == "Storm_Cell":
    weights_path = weightpath+'/mask_rcnn_nucleus_cell.h5' 
print("weights_path : ", weights_path)

config = NucleusInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=weights_path)
model.load_weights(weights_path, by_name=True)
result = model.detect([img_hoechst], verbose=0)[0]

mask, bbox, class_ids = result["masks"], result["rois"], result["class_ids"]
print("Origin shape: ", img_origin.shape, flush=True)
log("image", img_hoechst)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
print("##########################################################", flush=True)

# 4. post processing
print("4. post processing", flush=True)
## 4.1 remove outliers by roundness
print("# 4.1 remove outliers by roundness", flush=True)
def compute_roundness(label_image):
    contours, hierarchy = cv2.findContours(np.array(label_image, dtype=np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(contours[0]) * 4 * math.pi
    b = math.pow(cv2.arcLength(contours[0], True), 2)
    if b == 0:
        return 0
    return a / b

total_roundness=[]
for i in range(bbox.shape[0]):
    mmask = mask[:,:,i]
    total_roundness.append(compute_roundness(mmask))
print("len(total_roundness): ",len(total_roundness), flush=True)
roundness_outliers = []
roundness_zscore = np.abs(stats.zscore(total_roundness))
for c in range(bbox.shape[0]):
    roundness_outliers.append((total_roundness[c])<0.7)
print("roundness_outliers: ", flush=True)
for c in range(len(total_roundness)):
    if roundness_outliers[c]:
        print((roundness_outliers[c], c, total_roundness[c], roundness_zscore[c]), flush=True)

print("# 4.2 remove outliers by masks_area", flush=True)
total_masks_area=[]
for i in range(bbox.shape[0]):
    total_masks_area.append(np.sum(mask[:,:,i]))
print("len(total_masks_area): ",len(total_masks_area))
masks_outliers = []
masks_zscore = np.abs(stats.zscore(total_masks_area))
for c in range(bbox.shape[0]):
    masks_outliers.append((total_masks_area[c])<30000)
print("masks_outliers: ", flush=True)
for c in range(len(total_masks_area)):
    if masks_outliers[c]:
        print((masks_outliers[c], c, total_masks_area[c], masks_zscore[c]), flush=True)

print("# 4.3 remove outliers by boxes_aspect", flush=True)
total_boxes_size=[]
for i in range(bbox.shape[0]):
    box = bbox[i]
    total_boxes_size.append((box[2]-box[0])/(box[3]-box[1]))
print("len(total_boxes_size): ",len(total_boxes_size))
boxes_outliers = []
boxes_zscore = np.abs(stats.zscore(total_boxes_size))
for zs in boxes_zscore:
    boxes_outliers.append((zs>=1.5))
print("boxes_outliers: ", flush=True)     
for c in range(len(total_boxes_size)):
    if boxes_outliers[c]:
        print((boxes_outliers[c], c, total_boxes_size[c], boxes_zscore[c]), flush=True)

print("# 4.4 All outliers together", flush=True)
outliers=[]
for i in range(len(masks_outliers)): 
    outliers.append(masks_outliers[i] or roundness_outliers[i] or boxes_outliers[i])
F_total_boxes=[]
F_total_masks=[]
for cc in range(len(outliers)):
    if not outliers[cc]:
        F_total_boxes.append(bbox[cc])
        F_total_masks.append(mask[:,:,cc])
F_total_boxes = np.array(F_total_boxes)
F_total_masks=np.transpose(np.array(F_total_masks),(1,2,0))
F_total_class_ids=np.ones(F_total_boxes.shape[0], dtype=np.int32)
print("total_boxes.shape: ",F_total_boxes.shape, flush=True)
print("total_masks.shape: ",F_total_masks.shape, flush=True)

visualize.display_instances_save(img_hoechst, F_total_boxes, F_total_masks, F_total_class_ids, ["BG","nucleus"], 
                                     savename=sys.argv[5]+"/"+imagename[:-8]+".png", figsize=(20, 20))
print("save maskrcnn segmentation results as ", sys.argv[5]+"/"+imagename[:-8]+".png", flush=True)
print("##########################################################", flush=True)
print("5. Split and save each cell", flush=True)

for nn in range(F_total_boxes.shape[0]):
    bb = F_total_boxes[nn]
    mmask = F_total_masks[:,:,nn][bb[0]:bb[2], bb[1]:bb[3]].astype(np.uint16)
    mmask = cv2.merge([mmask,mmask,mmask])
    iimg = img_origin[bb[0]:bb[2], bb[1]:bb[3]]
    iimg = np.multiply(iimg, mmask)

    savename=imagename[:-8]+"_"+str(nn)+'.tif'
    cv2.imwrite(savepath+"/"+savename, iimg)
    print("save name with ", savepath+"/"+savename, flush=True)