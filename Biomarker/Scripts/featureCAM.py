print("🥁 PYTHON SCRIPT START", flush=True)
print("🚀 0. Import Libraries and Mount Drive", flush=True)
import histomicstk.features as hf
import numpy as np
import cv2, argparse
import pandas as pd
from skimage.measure import label
import sys
sys.path.append('/groups/4/gaa50089/acd13264yb/Epigenetic/Classification/')
from Scripts.utils import nucleus_intensity_distribution
print("done", flush=True)
print("##########################################################", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl_type", type=str, default="CTRL")
    parser.add_argument("--stain_type", type=str, default="All")
    parser.add_argument("--cam_type", type=str, default="GradCAM")
    parser.add_argument("--image_path", type=str, default="/home/acd13264yb/DDDog/Epigenetic/Classification/results_cam")
    parser.add_argument("--save_path", type=str, default="/home/acd13264yb/DDDog/Epigenetic/Features")
    args = parser.parse_args()
ctrl_type = args.ctrl_type
stain_type = args.stain_type
cam_type = args.cam_type
image_path = args.image_path
save_path = args.save_path
loadname = f"{ctrl_type}_{stain_type}_Resnet10_noavg_{cam_type}"
print(f"loadname: {loadname}")

print("🚀 1. Load Images and Compute Masks", flush=True)
img_all = np.load(f"{image_path}/{loadname}/{loadname}_img.npy",allow_pickle=True)
cam_all = np.load(f"{image_path}/{loadname}/{loadname}_cam.npy",allow_pickle=True)
print(f"img all shape: {img_all.shape}", flush=True)
print(f"cam all shape: {cam_all.shape}", flush=True)

mask_all = []
for img in img_all:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
    mask_all.append(thresh)
mask_all = np.array(mask_all).astype(np.uint8)
print(f"mask all shape: {mask_all.shape}", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
labels = [
    "Intensity.wholeNucleus",
    "Intensity.part05", 
    "Intensity.part04", 
    "Intensity.part03", 
    "Intensity.part02", 
    "Intensity.part01",
    "Intensity.distribution.part05", 
    "Intensity.distribution.part04", 
    "Intensity.distribution.part03", 
    "Intensity.distribution.part02", 
    "Intensity.distribution.part01"
]
print("🚀 2. Compute and Save nuclei features", flush=True)


# Init DataFrame to save
features_all = pd.DataFrame()
# Compute and Save nuclei features
for i in range(len(cam_all)):
    im_label = mask_all[i]
    im_nuclei = cam_all[i]
    features = hf.compute_nuclei_features(im_label=im_label, im_nuclei=im_nuclei)
    features["Label"] = i
    # Add new feature intensity distribution part 5 ~ 0 to DataFrame 
    intensity_distribution = nucleus_intensity_distribution(im_label, im_nuclei)
    for part, label in enumerate(labels):
        features[label] = intensity_distribution[part]
    # 合并 features 到 features_all
    features_all = pd.concat([features_all, features], ignore_index=True)
# Save DataFrame as CSV
features_name = f'{save_path}/features_{loadname}.csv'
features_all.to_csv(features_name, index=False)
print(f"🔥 Save features_all as {features_name}", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)