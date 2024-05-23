print("🥁 PYTHON SCRIPT START", flush=True)
print("🚀 0. Import Libraries and Mount Drive", flush=True)
import cv2,os,argparse
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stain_type", type=str, default="H3K27ac")
    parser.add_argument("--ctrl_type", type=str, default="RETT")
    parser.add_argument("--model_type", type=str, default="Resnet10_noavg")
    parser.add_argument("--image_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/Datasets")
    parser.add_argument("--model_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/results")
    parser.add_argument("--cam_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/results_cam")
    parser.add_argument("--cam_type", type=str, default="GradCAM")
    parser.add_argument("--target_layer", type=str, default="model.resnet.layer2")
    args = parser.parse_args()
stain_type = args.stain_type
ctrl_type = args.ctrl_type
model_type = args.model_type
image_path = args.image_path
model_path = args.model_path
cam_path = args.cam_path
cam_type = args.cam_type
target_layer = args.target_layer
# 1. Define GradCAM
print("🚀 1. Define GradCAM", flush=True)
def predict(model,input_tensor,true_y):
    model.eval()
    output = model(input_tensor)
    pre_y = output.argmax(1).cpu().item()
    return pre_y==true_y, pre_y

def gradcams(model,input_tensor,target_layers,imgpad,cam_type):
    # GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    if cam_type=="GradCAM":
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="ScoreCAM":
        cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="GradCAMPlusPlus":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="AblationCAM":
        cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="XGradCAM":
        cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="EigenCAM":
        cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
    elif cam_type=="FullGrad":
        cam = FullGrad(model=model, target_layers=target_layers, use_cuda=True)
    else:
        return print("your option is not support")
    targets = None
    grayscale_cam = cam(input_tensor, targets)[0]
    visualization = show_cam_on_image(imgpad, grayscale_cam, use_rgb=False)
    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    return visualization, grayscale_cam

print("done", flush=True)
print("##########################################################", flush=True)

# 2. Load Data and Model
print("🚀 2. Load Data and Model", flush=True)
X = np.load(f"{image_path}/{ctrl_type}_{stain_type}.npy", allow_pickle=True)
print(f"x.shape = {X.shape}", flush=True)

fold = 0
loadmodel = f"{model_path}/{model_type}_{stain_type}/Fold{str(fold)}.pkl"
if model_type=="Resnet10_noavg":
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet,self).__init__()
            self.resnet = models.resnet18(weights=True)
            self.resnet.layer3 = nn.Sequential()
            self.resnet.layer4 = nn.Sequential()
            self.resnet.avgpool = nn.Sequential()
            self.resnet.fc = nn.Linear(128*75*75, 2)
            self.resnet.load_state_dict(torch.load(loadmodel))   
        def forward(self, x):
            x = self.resnet(x)
            x = nn.Softmax(dim=1)(x)
            return x
elif model_type=="Resnet10":
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet,self).__init__()
            self.resnet = models.resnet18(weights=True)
            self.resnet.layer3 = nn.Sequential()
            self.resnet.layer4 = nn.Sequential()
            self.resnet.fc = nn.Linear(128, 2)
            self.resnet.load_state_dict(torch.load(loadmodel))
        def forward(self, x):
            x = self.resnet(x)
            # x = nn.Softmax(dim=1)(x)
            return x
elif model_type=="Resnet18":
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet,self).__init__()
            self.resnet = models.resnet18(weights=True)
            self.resnet.fc = nn.Linear(512, 2)
            self.resnet.load_state_dict(torch.load(loadmodel))
        def forward(self, x):
            x = self.resnet(x)
            # x = nn.Softmax(dim=1)(x)
            return x
device = "cuda"    
model = ResNet().to(device)
print("done", flush=True)
print("##########################################################", flush=True)

total = len(X)
# 3. SAVE CAM heatmap in all data
print("🚀 3. SAVE CAM heatmap ", flush=True)
if ctrl_type == "Ctrl": true_y = 0
elif ctrl_type == "VPA": true_y = 1
if target_layer=="model.resnet.layer2": target_layer = model.resnet.layer2
savename = f"{ctrl_type}_{stain_type}_{model_type}_{cam_type}"
count = 0
all_cam = []
all_img = []
for n in range(total):
    img = X[n]
    img = np.float32(img)/255
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)
    tf,pred = predict(model,input_tensor,true_y)
    if tf == True:
        count+=1
        all_img.append(img)
        visualization, cam = gradcams(model,input_tensor,[target_layer],img,cam_type)
        all_cam.append(cam)
        cv2.imwrite(f"{cam_path}/{savename}_{count:04}.jpg",visualization)
all_cam = np.array(all_cam)
all_img = np.array(all_img)
print("🌺 all_img.shape: ", all_img.shape, flush=True)
print("🌺 all_cam.shape: ", all_cam.shape, flush=True)
print("🌺 Data accuracy: {:.3f}, count: {:}, total: {:} ".format(count/total,count,total), flush=True)

np.save(f"{cam_path}/{savename}_img.npy", all_img)
np.save(f"{cam_path}/{savename}_cam.npy", all_cam)
print(f"🌺 Save all cam heatmap to {cam_path}", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)