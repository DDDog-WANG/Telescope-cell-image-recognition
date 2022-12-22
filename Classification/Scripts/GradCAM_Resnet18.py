print("##########################################################", flush=True)

print("0. Import Libraries and Mount Drive", flush=True)
import cv2,os,sys
from skimage import io
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
print("done", flush=True)
print("##########################################################", flush=True)

# 1. Define GradCAM
print("1. Define GradCAM", flush=True)
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image
from torchvision.models import resnet50
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
print("done", flush=True)
print("##########################################################", flush=True)

# 2. Load Data and Model
resnet=sys.argv[1]
chip=sys.argv[2]
homepath = sys.argv[3]
savepath=sys.argv[4]
print(chip, flush=True)
print(resnet, flush=True)
print(homepath, flush=True)
print(savepath, flush=True)
# Load Data
print("2. Load Data and Model", flush=True)
X_Ctrl = np.load(homepath+"/Datasets/Ctrl_"+chip+".npy",allow_pickle=True)
X_VPA = np.load(homepath+"/Datasets/VPA_"+chip+".npy",allow_pickle=True)
y_Ctrl = torch.zeros(len(X_Ctrl), dtype=torch.int64)
y_VPA = torch.ones(len(X_VPA), dtype=torch.int64)
# load model
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.resnet = models.resnet18(weights=True)
        self.resnet.fc = nn.Linear(512, 2)
        self.resnet.load_state_dict(torch.load(homepath+"/Models/"+resnet+"_"+chip+".pkl"))
    def forward(self, x):
        x = self.resnet(x)
        x = nn.Softmax(dim=1)(x)
        return x
device = "cuda"
model = ResNet().to(device)
print("done", flush=True)
print("##########################################################", flush=True)

# 3. GradCAM heatmap in samples 
print("3. GradCAM heatmap in samples", flush=True)
target_layers = [model.resnet.layer4[1].conv2]
# cam's option: GradCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, ScoreCAM, FullGrad
optioncam = "GradCAM"
print("3.1 GradCAM heatmap in samples of Ctrl", flush=True)
true_y = 0
fig, ax = plt.subplots(5,5, figsize = (40,40))
for n in range(25):
    img = X_Ctrl[n]
    img = np.float32(img)/255
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)
    tf,pred = predict(model,input_tensor,true_y)
    visualization,gb,cam_gb,cam = gradcams(model,input_tensor,target_layers,img,optioncam)
    plt.subplot(5,5,n+1)
    plt.title(tf)
    plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
savename = savepath+"/GradCAM_sample_"+resnet+"_Ctrl_"+chip+".png"
plt.savefig(savename)
print("Save png as "+savename, flush=True)
print("done", flush=True)
print("3.2 GradCAM heatmap in samples of VPA", flush=True)
true_y = 1
fig, ax = plt.subplots(5,5, figsize = (40,40))
times=0
for n in range(50,200):
    img = X_VPA[n]
    img = np.float32(img)/255
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)
    tf,pred = predict(model,input_tensor,true_y)
    visualization,gb,cam_gb,cam = gradcams(model,input_tensor,target_layers,img,optioncam)
    if np.amax(cam)!=0:
        if times < 25:
            times+=1
            plt.subplot(5,5,times)
            plt.title(tf)
            plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
        else:
            break
savename = savepath+"/GradCAM_sample_"+resnet+"_VPA_"+chip+".png"
plt.savefig(savename)
print("Save png as "+savename, flush=True)
print("done", flush=True)
print("##########################################################", flush=True)

# 4. Average Gradcam heatmap in all data
print("4. Average Gradcam heatmap in all data", flush=True)
def rotate(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    rotated = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(img, rotated, (w, h))
    return rotated
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
def rectangle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0 , 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h
total = 3000
print("4.1 Average Gradcam heatmap in Ctrl", flush=True)
true_y = 0
count = 0
intensity00_conv1 = np.zeros((600,600))
intensity00_layer1 = np.zeros((600,600))
intensity00_layer2 = np.zeros((600,600))
intensity00_layer3 = np.zeros((600,600))
intensity00_layer4 = np.zeros((600,600))
intensity00_gb = np.zeros((600,600))
for n in tqdm(range(total)):
    img = X_Ctrl[n]
    result, angle = ellipse(img)
    x, y, w, h = rectangle(result)
    img = np.float32(img)/255
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)
    tf,pred = predict(model,input_tensor,true_y)
    if tf == True:
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.conv1],img,optioncam)
        intensity00_conv1+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer1[1].conv2],img,optioncam)
        intensity00_layer1+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer2[1].conv2],img,optioncam)
        intensity00_layer2+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer3[1].conv2],img,optioncam)
        intensity00_layer3+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer4[1].conv2],img,optioncam)
        intensity00_layer4+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        intensity00_gb+=ZeroPaddingResizeCV(rotate(gb[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        count+=1
print("Cell in Ctrl acc: {:.3f}, count: {:}, total: {:} " .format(count/total,count,total), flush=True)
intensity00_conv1 = intensity00_conv1/(count*np.amax(intensity00_conv1))
intensity00_layer1 = intensity00_layer1/(count*np.amax(intensity00_layer1))
intensity00_layer2 = intensity00_layer2/(count*np.amax(intensity00_layer2))
intensity00_layer3 = intensity00_layer3/(count*np.amax(intensity00_layer3))
intensity00_layer4 = intensity00_layer4/(count*np.amax(intensity00_layer4))
intensity00_gb = intensity00_gb/(count*np.amax(intensity00_gb))
plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.subplot(1,6,1)
plt.axis('off')
plt.title("GradCAM: conv1")
plt.imshow(intensity00_conv1)
plt.subplot(1,6,2)
plt.axis('off')
plt.title("GradCAM: layer1")
plt.imshow(intensity00_layer1)
plt.subplot(1,6,3)
plt.axis('off')
plt.title("GradCAM: layer2")
plt.imshow(intensity00_layer2)
plt.subplot(1,6,4)
plt.axis('off')
plt.title("GradCAM: layer3")
plt.imshow(intensity00_layer3)
plt.subplot(1,6,5)
plt.axis('off')
plt.title("GradCAM: layer4")
plt.imshow(intensity00_layer4)
plt.subplot(1,6,6)
plt.axis('off')
plt.title("guidedprop")
plt.imshow(intensity00_gb)
savename = savepath+"/GradCAM_average_"+resnet+"_Ctrl_"+chip+".png"
plt.savefig(savename)
print("Save png as "+savename, flush=True)
print("done", flush=True)

print("4.2 Average Gradcam heatmap in VPA", flush=True)
true_y = 1
count = 0
intensity01_conv1 = np.zeros((600,600))
intensity01_layer1 = np.zeros((600,600))
intensity01_layer2 = np.zeros((600,600))
intensity01_layer3 = np.zeros((600,600))
intensity01_layer4 = np.zeros((600,600))
intensity01_gb = np.zeros((600,600))
for n in tqdm(range(total)):
    img = X_VPA[n]
    result, angle = ellipse(img)
    x, y, w, h = rectangle(result)
    img = np.float32(img)/255
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)
    tf,pred = predict(model,input_tensor,true_y)
    if tf == True:
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.conv1],img,optioncam)
        intensity01_conv1+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer1[1].conv2],img,optioncam)
        intensity01_layer1+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer2[1].conv2],img,optioncam)
        intensity01_layer2+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer3[1].conv2],img,optioncam)
        intensity01_layer3+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        visualization,gb,cam_gb,cam = gradcams(model,input_tensor,[model.resnet.layer4[1].conv2],img,optioncam)
        intensity01_layer4+=ZeroPaddingResizeCV(rotate(cam[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        intensity01_gb+=ZeroPaddingResizeCV(rotate(gb[:,:,0], angle-45)[y:y+h,x:x+w], size=(600, 600),n=1)
        count+=1
print("Cell in VPA acc: {:.3f}, count: {:}, total: {:} " .format(count/total,count,total), flush=True)
intensity01_conv1 = intensity01_conv1/(count*np.amax(intensity01_conv1))
intensity01_layer1 = intensity01_layer1/(count*np.amax(intensity01_layer1))
intensity01_layer2 = intensity01_layer2/(count*np.amax(intensity01_layer2))
intensity01_layer3 = intensity01_layer3/(count*np.amax(intensity01_layer3))
intensity01_layer4 = intensity01_layer4/(count*np.amax(intensity01_layer4))
intensity01_gb = intensity01_gb/(count*np.amax(intensity01_gb))
plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.subplot(1,6,1)
plt.axis('off')
plt.title("GradCAM: conv1")
plt.imshow(intensity01_conv1)
plt.subplot(1,6,2)
plt.axis('off')
plt.title("GradCAM: layer1")
plt.imshow(intensity01_layer1)
plt.subplot(1,6,3)
plt.axis('off')
plt.title("GradCAM: layer2")
plt.imshow(intensity01_layer2)
plt.subplot(1,6,4)
plt.axis('off')
plt.title("GradCAM: layer3")
plt.imshow(intensity01_layer3)
plt.subplot(1,6,5)
plt.axis('off')
plt.title("GradCAM: layer4")
plt.imshow(intensity01_layer4)
plt.subplot(1,6,6)
plt.axis('off')
plt.title("guidedprop")
plt.imshow(intensity01_gb)
savename = savepath+"/GradCAM_average_"+resnet+"_VPA_"+chip+".png"
plt.savefig(savename)
print("Save png as "+savename, flush=True)
print("done", flush=True)