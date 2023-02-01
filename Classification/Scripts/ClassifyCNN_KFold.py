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
print("done", flush=True)
print("##########################################################", flush=True)

print("3. Develop model", flush=True)
if restype=="Resnet10_noavg":
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
elif restype=="Resnet10":
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet,self).__init__()
            self.resnet = models.resnet18(weights=True)
            self.resnet.layer3 = nn.Sequential()
            self.resnet.layer4 = nn.Sequential()
            self.resnet.fc = nn.Linear(128, 2)   
        def forward(self, x):
            x = self.resnet(x)
            x = nn.Softmax(dim=1)(x)
            return x
elif restype=="Resnet18":
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet,self).__init__()
            self.resnet = models.resnet18(weights=True)
            self.resnet.fc = nn.Linear(512, 2)   
        def forward(self, x):
            x = self.resnet(x)
            x = nn.Softmax(dim=1)(x)
            return x    
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
n_splits=5
splits=KFold(n_splits,shuffle=True,random_state=42)
batch_size = 128
n_epochs = 100
history = {'loss_train': [], 'loss_valid': [],'acc_train':[],'acc_valid':[]}
for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold + 1), flush=True)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    dataloader_train = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    dataloader_valid = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    model = ResNet().to(device)
    ngpu = 4
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    for epoch in range(n_epochs):
        loss_train, acc_train = train(model,device,dataloader_train,loss_function,optimizer)
        loss_valid, acc_valid = valid(model,device,dataloader_valid,loss_function)
        scheduler.step()
        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'
              .format(epoch, loss_train, acc_train, loss_valid, acc_valid), flush=True)
        history['loss_train'].append(loss_train)
        history['loss_valid'].append(loss_valid)
        history['acc_train'].append(acc_train)
        history['acc_valid'].append(acc_valid)
    savemodel = homepath+"/Models/"+restype+"_Fold"+str(fold)+"_"+datatype+".pkl"
    for param in model.parameters():
        param.requires_grad = True
        torch.save(model.module.resnet.state_dict(),savemodel)
    print("saved model as "+savemodel, flush=True)
print("done", flush=True)
print("##########################################################", flush=True)
homepath
print("6. plot the figure of training process", flush=True)
loss_train_avg = np.zeros(n_epochs)
loss_valid_avg = np.zeros(n_epochs)
acc_train_avg = np.zeros(n_epochs)
acc_valid_avg = np.zeros(n_epochs)
for num in range(n_splits):
    loss_train = history['loss_train'][num*n_epochs:(num+1)*n_epochs]
    loss_valid = history['loss_valid'][num*n_epochs:(num+1)*n_epochs]
    acc_train = history['acc_train'][num*n_epochs:(num+1)*n_epochs]
    acc_valid = history['acc_valid'][num*n_epochs:(num+1)*n_epochs]
    loss_train_avg+=loss_train
    loss_valid_avg+=loss_valid
    acc_train_avg+=acc_train
    acc_valid_avg+=acc_valid
    name_title="Fold_"+str(num)+'_Training and Validation accuracy'
    savepath = homepath+"/results/"+restype+"_"+datatype+"/"+name_title+".png"
    plt.figure(figsize=(12, 8))
    plt.ylim(0,1.0)
    plt.plot(range(1,n_epochs+1), acc_train, 'b', label='Training accuracy')  
    plt.plot(range(1,n_epochs+1), acc_valid, 'r', label='Validation accuracy')
    plt.title(name_title)
    plt.savefig(savepath)
    print("Saved output as , ", savepath, flush=True)
loss_train_avg = loss_train_avg/n_splits
loss_valid_avg = loss_valid_avg/n_splits
acc_train_avg = acc_train_avg/n_splits
acc_valid_avg = acc_valid_avg/n_splits
name_title="Average Training and Validation accuracy"
savepath = homepath+"/results/"+restype+"_"+datatype+"/"+name_title+".png"
plt.figure(figsize=(12, 8))
plt.ylim(0,1.0)
plt.plot(range(1,n_epochs+1), acc_train_avg, 'b', label='Training accuracy')  
plt.plot(range(1,n_epochs+1), acc_valid_avg, 'r', label='Validation accuracy')
plt.title(name_title)
plt.savefig(savepath)
print("Saved output as , ", savepath, flush=True)
print("##########################################################", flush=True)
print(history, flush=True)
print("done", flush=True)