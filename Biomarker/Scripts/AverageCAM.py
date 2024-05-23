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
# 1. Define CAM
print("1. Define CAM", flush=True)
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
    return visualization,gb,cam_gb,cam_mask

def ellipse(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0 , 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(big_contour)
    (xc,yc),(d1,d2),angle = ellipse
    result = img.copy()
    cv2.ellipse(result, ellipse, (0, 255, 0), 3)
    xc, yc = ellipse[0]
    cv2.circle(result, (int(xc),int(yc)), 10, (255, 255, 255), -1)
    rmajor = max(d1,d2)/2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    xtop = xc + math.cos(math.radians(angle))*rmajor
    ytop = yc + math.sin(math.radians(angle))*rmajor
    xbot = xc + math.cos(math.radians(angle+180))*rmajor
    ybot = yc + math.sin(math.radians(angle+180))*rmajor
    cv2.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 3)
    return result, angle

def rectangle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0 , 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h

def rotate(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    rotated = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(img, rotated, (w, h))
    return rotated

def ZeroPaddingResizeCV(img, size=(600, 600), interpolation=None, n=3):
    isize = img.shape
    ih, iw = isize[0], isize[1]
    h, w = size[0], size[1]
    scale = min(w / iw, h / ih)
    new_w = int(iw * scale + 0.5)
    new_h = int(ih * scale + 0.5)
 
    img = cv2.resize(img, (new_w, new_h), interpolation)
    if n==3:
        new_img = np.zeros((h, w, n), np.uint8)
        new_img[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2] = img
    else:
        new_img = np.zeros((h, w), np.float32)
        new_img[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2] = img
    return new_img
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

print("3. Average cam heatmap ", flush=True)
if drag == "Ctrl": true_y = 0
elif drag == "VPA": true_y = 1
count = 0
intensity00_conv1 = []
intensity00_layer1 = []
intensity00_layer2 = []
intensity00_layer3 = []
intensity00_layer4 = []
intensity00_gb = []
for n in tqdm(range(total)):
    img = X[n]
    result, angle = ellipse(img)
    x, y, w, h = rectangle(result)
    img = np.float32(img)/255
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)
    tf,pred = predict(model,input_tensor,true_y)
    if tf == True:
        count+=1
        # conv1
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.conv1],img,optioncam)
        heatmap = ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
#         heatmap[heatmap<0.5]=0
        intensity00_conv1.append(heatmap)
        # layer1
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer1],img,optioncam)
        heatmap = ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
#         heatmap[heatmap<0.5]=0
        intensity00_layer1.append(heatmap)
        # layer2
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer2],img,optioncam)
        heatmap = ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
#         heatmap[heatmap<0.5]=0
        intensity00_layer2.append(heatmap)
        # gb
        heatmap = ZeroPaddingResizeCV(rotate(gb[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
#         heatmap[heatmap<0.5]=0
        intensity00_gb.append(heatmap)
intensity00_conv1 = np.array(intensity00_conv1)
intensity00_layer1 = np.array(intensity00_layer1)
intensity00_layer2 = np.array(intensity00_layer2)
intensity00_gb = np.array(intensity00_gb)
print("intensity00_layer2.shape: ", intensity00_layer2.shape, flush=True)
print("Data00 acc: {:.3f}, count: {:}, total: {:} " .format(count/total,count,total), flush=True)
np.save(dir+"/Datasets/"+optioncam+"_"+drag+"_"+chip+"_conv1.npy" ,intensity00_conv1)
np.save(dir+"/Datasets/"+optioncam+"_"+drag+"_"+chip+"_layer1.npy" ,intensity00_layer1)
np.save(dir+"/Datasets/"+optioncam+"_"+drag+"_"+chip+"_layer2.npy" ,intensity00_layer2)
np.save(dir+"/Datasets/"+optioncam+"_"+drag+"_"+chip+"_gb.npy" ,intensity00_gb)
print("Save all cam heatmap to .npy" ,flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
print("##########################################################", flush=True)
print("##########################################################", flush=True)