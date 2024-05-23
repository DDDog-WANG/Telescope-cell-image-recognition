print("##########################################################", flush=True)

print("0. Import Libraries and Mount Drive", flush=True)
import cv2,os,sys
import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image
from torchvision.models import resnet50
print("done", flush=True)
print("##########################################################", flush=True)
print(sys.argv[1], flush=True)
print(sys.argv[2], flush=True)
print(sys.argv[3], flush=True)
print(sys.argv[4], flush=True)
dir = sys.argv[3]
# 1. Define GradCAM
print("1. Define GradCAM", flush=True)
def predict(model,input_tensor,true_y):
    model.eval()
    output = model(input_tensor)
    pre_y = output.argmax(1).cpu().item()
    return pre_y==true_y, pre_y

def gradcams(model,input_tensor,target_layers,imgpad,optioncam):
    # GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    if optioncam=="GradCAM":
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif optioncam=="ScoreCAM":
        cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif optioncam=="GradCAMPlusPlus":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    elif optioncam=="AblationCAM":
        cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif optioncam=="XGradCAM":
        cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif optioncam=="EigenCAM":
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif optioncam=="FullGrad":
        cam = FullGrad(model=model, target_layers=target_layers, use_cuda=True)
    else:
        return print("your option is not support")
    targets = None
    grayscale_cam = cam(input_tensor, targets)[0]
    visualization = show_cam_on_image(imgpad, grayscale_cam, use_rgb=False)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_tensor, targets)
    if np.amax(gb)!=0:
        gb = np.maximum(gb,0)*(1/np.amax(gb))
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = cam_mask * gb
    return visualization,grayscale_cam,cam_gb,cam_mask

def pltcam(img, y, chip):
    #     model.resnet.load_state_dict(torch.load("./Models/"+resnet+"_"+chip+"/Fold0.pkl"))
    img = np.float32(img)/255
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)
    tf,pred = predict(model,input_tensor,y)
    visualization,cam_one,cam_gb,cam_merge = gradcams(model,input_tensor,target_layers,img,optioncam)
    return cam_one
print("done", flush=True)
print("##########################################################", flush=True)

# 2. Load Data and Model
print("2. Load Data and Model", flush=True)
chip = sys.argv[1]
optioncam = sys.argv[2]
drag = sys.argv[4]
resnet="Resnet10_noavg"
X = np.load(dir+"/Datasets/"+drag+"_"+chip+".npy",allow_pickle=True)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.resnet = models.resnet18(weights=True)
        self.resnet.layer3 = nn.Sequential()
        self.resnet.layer4 = nn.Sequential()
        self.resnet.avgpool = nn.Sequential()
        self.resnet.fc = nn.Linear(128*75*75, 2)
    def forward(self, x):
        x = self.resnet(x)
        x = nn.Softmax(dim=1)(x)
        return x
device = "cuda"    
model = ResNet().to(device)
model.resnet.load_state_dict(torch.load(dir+"/Models/"+resnet+"_"+chip+"/Fold0.pkl"))
print("done", flush=True)
print("##########################################################", flush=True)

total = 3000
# 3. Average cam heatmap in all data
target_layers = [model.resnet.layer2]
print("3. Average cam heatmap ", flush=True)
if drag == "Ctrl": true_y = 0
elif drag == "VPA": true_y = 1
count = 0
num_parts = 5
intensity=[]
all_part_intensity=[]
for n in tqdm(range(total)):
    img = X[n]
    img = np.float32(img)/255
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)
    tf,pred = predict(model,input_tensor,true_y)
    if tf == True:
        count+=1
        visualization,cam_img,cam_gb,cam_merge = gradcams(model,input_tensor,target_layers,img,optioncam)
        # intensity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0 , 1, cv2.THRESH_BINARY)[1]
        cam_mask = cam_img * thresh
        non_zero_values = cam_mask[np.nonzero(cam_mask)]
        average = np.mean(non_zero_values)
        intensity.append(average)
        # part_distribution_intensity
        mask = thresh.copy()
        h, w = thresh.shape[:2]
        num_parts = 5
        for i in range(num_parts, 0, -1):
            scale = i/5
            new_h, new_w = int(h * scale), int(w * scale)
            d_h, d_w = (h - new_h) // 2, (w - new_w) // 2
            resized = cv2.resize(thresh, (new_w, new_h))
            copy = np.zeros_like(thresh)
            copy[d_h:d_h + new_h, d_w:d_w + new_w] = resized*2
            copy[copy == 0] = 1
            mask *= copy
        for i in range(1,num_parts+1):
            mask[mask==2**i]=(255*i/num_parts)
        part_intensity=[]
        for i in range(1,num_parts+1):
            color = int(255*i/num_parts)
            mask_part = cv2.inRange(mask, np.array(color, dtype=np.uint8), np.array(color, dtype=np.uint8))/255
            cam_mask_part = cam_mask * mask_part
            non_zero_values = cam_mask_part[np.nonzero(cam_mask_part)]
            average = np.mean(non_zero_values)
            part_intensity.append(average)
        all_part_intensity.append(part_intensity)

intensity = np.array(intensity)
all_part_intensity = np.array(all_part_intensity)
print("intensity.shape: ", intensity.shape, flush=True)
print("all_part_intensity.shape: ", all_part_intensity.shape, flush=True)
print("Data00 acc: {:.3f}, count: {:}, total: {:} " .format(count/total,count,total), flush=True)
np.save(dir+"/Datasets/"+optioncam+"_"+drag+"_"+chip+"_intensity.npy" ,intensity)
np.save(dir+"/Datasets/"+optioncam+"_"+drag+"_"+chip+"_allpartintensity.npy" ,all_part_intensity)
print("Save all to .npy" ,flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
print("##########################################################", flush=True)
print("##########################################################", flush=True)