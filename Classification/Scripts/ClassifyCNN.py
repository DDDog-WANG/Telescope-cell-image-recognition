print("##########################################################", flush=True)

# 0. Import Libraries and Mount Drive
print("0. Import Libraries and Mount Drive", flush=True)
import cv2,os,sys
from skimage import io
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,precision_score,accuracy_score,roc_curve
import torch
from torch.utils.data import random_split,Dataset,DataLoader,SubsetRandomSampler
from torch.utils.data import Dataset,TensorDataset,random_split,SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("torch.device(cuda)", flush=True)
    print("torch.cuda.device_count(): ", torch.cuda.device_count(), flush=True)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(), flush=True)
    print("torch.cuda.current_device()", torch.cuda.current_device(), flush=True)
else:
    device = torch.device("cpu")
    print("torch.device(cpu)", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
homepath = sys.argv[1]
datatype = sys.argv[2]
restype = sys.argv[3]
print("This is "+restype, flush=True)
print(datatype, flush=True)
# 1. Load and Process Images
print("1. Load and Process Images", flush=True)
X_Ctrl = np.load(homepath+"/Datasets/Ctrl_"+datatype+".npy",allow_pickle=True)
X_VPA = np.load(homepath+"/Datasets/VPA_"+datatype+".npy",allow_pickle=True)
y_Ctrl = torch.zeros(len(X_Ctrl), dtype=torch.int64)
y_VPA = torch.ones(len(X_VPA), dtype=torch.int64)
X = np.concatenate((X_Ctrl, X_VPA), axis = 0)
y = torch.cat((y_Ctrl, y_VPA), 0)
class cell_dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.transform(self.x[idx]).to(torch.float), F.one_hot(self.y[idx],num_classes=2).to(torch.float)
dataset = cell_dataset(X, y)
batch_size = 128
train_size = int(len(X)*0.8)
valid_size = len(X) - train_size
train_data, valid_data = random_split(dataset=dataset,lengths=[train_size, valid_size],generator=torch.Generator().manual_seed(42))
dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
train_data_01 = 0
train_size = len(train_data)
for i in range(train_size):
    train_data_01+=int(train_data[i][1][1].item())
print("Total number of train : ", train_size, shuffle=True)
print("train_class_0: ", train_size-train_data_01, shuffle=True)
print("train_class_1: ", train_data_01, shuffle=True)
valid_data_01 = 0
valid_size = len(valid_data)
for i in range(valid_size):
    valid_data_01+=int(valid_data[i][1][1].item())
print("\nTotal number of valid : ", valid_size, shuffle=True)
print("valid_class_0 num : ", valid_size-valid_data_01, shuffle=True)
print("valid_class_1 num : ", valid_data_01, shuffle=True)
print("done", flush=True)
print("##########################################################", flush=True)

print("3. Develop model", flush=True)
if restype=="Resnet10":
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
elif restype=="Resnet18":
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet,self).__init__()
            self.resnet = models.resnet18(weights=True)
            self.resnet.avgpool = nn.Sequential()
            self.resnet.fc = nn.Linear(512*19*19, 2)   
        def forward(self, x):
            x = self.resnet(x)
            x = nn.Softmax(dim=1)(x)
            return x
model = ResNet().to(device)
ngpu = 4
if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))    
print("done", flush=True)
print("##########################################################", flush=True)

print("4. Define Training and Validation", flush=True)
def train(model,device,dataloader_train,loss_function,optimizer):
    losses_train = []
    n_train = 0
    acc_train = 0
    optimizer.step()
    model.train()
    for x, y in dataloader_train:
        n_train += y.size()[0]
        model.zero_grad()  # 勾配の初期化
        x = x.to(device)  # テンソルをGPUに移動
        y = y.to(device)
        output = model.forward(x)  # 順伝播
        loss = loss_function(output, y)  # 誤差(クロスエントロピー誤差関数)の計算
        loss.backward()  # 誤差の逆伝播
        optimizer.step()  # パラメータの更新
        acc_train += (output.argmax(1) == y[:,1]).float().sum().item()
        losses_train.append(loss.tolist())
    return np.mean(losses_train), (acc_train/n_train)       
def valid(model,device,dataloader_valid,loss_function):
    losses_valid = []
    n_val = 0
    acc_val = 0
    model.eval()
    for x, y in dataloader_valid:
        n_val += y.size()[0]
        x = x.to(device)  # テンソルをGPUに移動
        y = y.to(device)
        output = model.forward(x)  # 順伝播
        loss = loss_function(output, y)  # 誤差(クロスエントロピー誤差関数)の計算
        acc_val += (output.argmax(1) == y[:,1]).float().sum().item()
        losses_valid.append(loss.tolist())
    return np.mean(losses_valid), (acc_val/n_val)
print("done", flush=True)
print("##########################################################", flush=True)

print("5. Train by KFold of Cross Validation", flush=True)
n_epochs = 100
model.avgpool = nn.AdaptiveAvgPool2d(1)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
history = {'loss_train': [], 'loss_valid': [],'acc_train':[],'acc_valid':[]}
for epoch in range(n_epochs):
    loss_train, acc_train = train(model,device,dataloader_train,loss_function,optimizer)
    loss_valid, acc_valid = valid(model,device,dataloader_valid,loss_function)
    scheduler.step()
    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'
          .format(epoch, loss_train, acc_train, loss_valid, acc_valid))   
    history['loss_train'].append(loss_train)
    history['loss_valid'].append(loss_valid)
    history['acc_train'].append(acc_train)
    history['acc_valid'].append(acc_valid)
for param in model.parameters():
    param.requires_grad = True
    torch.save(model.module.resnet.state_dict(),homepath+"/Models/"+restype+"_"+datatype+".pkl")
print("saved model as "+homepath+"/Models/"+restype+"_"+datatype+".pkl", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
print("6. plot the figure of training process", flush=True)
name_title="Training and Validation accuracy"
savepath = homepath+"/results/"+restype+"_"+datatype+"/"+name_title+".png"
plt.figure(figsize=(12, 8))
plt.ylim(0,1.0)
plt.plot(range(1,n_epochs+1), history['acc_train'], 'b', label='Training accuracy')  
plt.plot(range(1,n_epochs+1), history['acc_valid'], 'r', label='Validation accuracy')
plt.title(name_title)
plt.savefig(savepath)
print("Saved output as , ", savepath, flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
print("5. Train by KFold of Cross Validation", flush=True)
y = []
y_pred = []
out_pred = []
total = len(valid_data)
model.eval()
for n in range(total):
    img = valid_data[n][0]
    label = valid_data[n][1][1].item()
    input_tensor = img.unsqueeze(0).to(device)
    output = model(input_tensor)
    pred = output.argmax(1).cpu().item()
    out_pred.append(output[0][1].item())
    y_pred.append(pred)
    y.append(label)
y = np.array(y)
y_pred = np.array(y_pred)
out_pred = np.array(out_pred)
print("total_test: {:}" .format(total), flush=True)
print('accuracy_score: {:.3f}'.format(accuracy_score(y,y_pred)), flush=True)
print('precision_score: {:.3f}'.format(precision_score(y,y_pred)), flush=True)
print('roc_auc_score: {:.3f}'.format(roc_auc_score(y, out_pred)), flush=True)
fpr, tpr, thresholds = roc_curve(y, out_pred,drop_intermediate=True)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
name_title="ROC curve"
savepath = homepath+"/results/"+restype+"_"+datatype+"/"+name_title+".png"
plt.savefig(savepath)
print("Saved output as , ", savepath, flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
print(history, flush=True)