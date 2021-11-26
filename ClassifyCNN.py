import sys
import cv2,os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import torchvision.models as models


# データの読み込み
path = sys.argv[1]
print("PATH is ", path)

DataPath=[]
DataLabel=[]

for root,dirs,files in os.walk(path):
    for file_name in files:
        path = os.path.join(root,file_name)
        label = os.path.join(file_name)
        DataPath.append(path)
        DataLabel.append(int(label[0])-1)
print("len(DataPath): ", len(DataPath))
print("len(DataLabel): ", len(DataLabel))


# データ処理

### 1. Max_shape_0 , Max_shape_1を調べる
DataSize=[]
shape_0=0
shape_1=0
for i in range(len(DataPath)):
    imgSize = cv2.imread(DataPath[i]).shape
    DataSize.append(imgSize)
    if imgSize[0]>shape_0:
        shape_0=imgSize[0]
    if imgSize[1]>shape_1:
        shape_1=imgSize[1]

        
print("Max_shape_0: ", shape_0)
print("Max_shape_1: ", shape_1)
print("")


### 2. 512*3に補正し、サイズを一致する
DataResize=[]

Max_shape_0 = 512*3
Max_shape_1 = 512*3

for path in DataPath:
    img = cv2.imread(path)
    imgSize = img.shape
    
    top_size,bottom_size = (Max_shape_0-imgSize[0])//2,(Max_shape_0-imgSize[0])//2
    left_size,right_size = (Max_shape_1-imgSize[1])//2,(Max_shape_1-imgSize[1])//2
    
    if (imgSize[0] % 2) != 0:
        top_size,bottom_size = (Max_shape_0-imgSize[0])//2,(Max_shape_0-imgSize[0])//2+1
        
    if (imgSize[1] % 2) != 0:     
        left_size,right_size = (Max_shape_1-imgSize[1])//2,(Max_shape_1-imgSize[1])//2+1
    
    img_pad = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=(0,0,0))
  
    DataResize.append(img_pad)

DataResize = np.array(DataResize)
print("DataResize: ", DataResize.shape)


### 3. bin_ndarrayによる512*512にresizeする
DataBinResize = []
def bin_ndarray(ndarray, new_shape, operation):
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

for img in DataResize:
    imgresize = bin_ndarray(img, new_shape=(512,512,3), operation='mean')
    DataBinResize.append(imgresize)


DataBinResize = np.array(DataBinResize)
print("DataBinResize: ", DataBinResize.shape)

img = cv2.imread(DataPath[1])
print("DataPath[1].shape: ", img.shape)
print("DataResize[1].shape: ", DataResize[1].shape)
print("DataBinResize[1].shape: ", DataBinResize[1].shape)


### ラベルを作る
data_img = DataBinResize
data_label=DataLabel
data_label=np.array(data_label)

print("data_img.shape: ", data_img.shape)
print("data_label.shape: ", data_label.shape)
print("data_label: ", data_label)


# data split
data_img_class=[[],[],[],[]]
data_label_class=[[],[],[],[]]

for i in range(len(data_label)):
    data_img_class[data_label[i]].append(data_img[i])
    data_label_class[data_label[i]].append(data_label[i])
    
X_train, X_test, y_train, y_test = [],[],[],[]

for i in range(len(data_img_class)):
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(np.array(data_img_class[i]), np.array(data_label_class[i]), test_size=0.3, random_state=42)        
    X_train.extend(X_train_class), X_test.extend(X_test_class), y_train.extend(y_train_class), y_test.extend(y_test_class)

X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# shuffle
train_permutation = np.random.permutation(X_train.shape[0])
test_permutation = np.random.permutation(X_test.shape[0])
X_train,y_train, = X_train[train_permutation,:,:,:], y_train[train_permutation] 
X_test,y_test = X_test[test_permutation,:,:,:], y_test[test_permutation]

print("Total number of train : ", len(y_train))
print("train_class_1 num : ", y_train.tolist().count(0))
print("train_class_2 num : ", y_train.tolist().count(1))
print("train_class_3 num : ", y_train.tolist().count(2))
print("train_class_4 num : ", y_train.tolist().count(3))
print("")
print("Total number of test : ", len(y_test))
print("test_class_1 num : ", y_test.tolist().count(0))
print("test_class_2 num : ", y_test.tolist().count(1))
print("test_class_3 num : ", y_test.tolist().count(2))
print("test_class_4 num : ", y_test.tolist().count(3))


# train data augmentation 
n = len(X_train)
X_augment=X_train
y_augment=y_train

for i in tqdm(range(n)):
    img = X_train[i]
    label = y_train[i]

    #水平镜像
    h_flip=cv2.flip(img,1)
    X_augment=np.concatenate((X_augment,h_flip[np.newaxis,:]),axis=0)
    y_augment=np.append(y_augment,label)
    
    #垂直镜像
    v_flip=cv2.flip(img,0)
    X_augment=np.concatenate((X_augment,v_flip[np.newaxis,:]),axis=0)
    y_augment=np.append(y_augment,label)
    
    #水平垂直镜像
    hv_flip=cv2.flip(img,-1)
    X_augment=np.concatenate((X_augment,hv_flip[np.newaxis,:]),axis=0)
    y_augment=np.append(y_augment,label)
    
    #平移矩阵[[1,0,-100],[0,1,-12]]
    M=np.array([[1,0,-100],[0,1,-12]],dtype=np.float32)
    translation=cv2.warpAffine(img,M,(512,512))
    X_augment=np.concatenate((X_augment,translation[np.newaxis,:]),axis=0)
    y_augment=np.append(y_augment,label)

    #45度旋转
    rows,cols=img.shape[:2]
    M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
    rotate_45=cv2.warpAffine(img,M,(cols,rows))
    X_augment=np.concatenate((X_augment,rotate_45[np.newaxis,:]),axis=0)
    y_augment=np.append(y_augment,label)
    
    #60度旋转
    rows,cols=img.shape[:2]
    M=cv2.getRotationMatrix2D((cols/2,rows/2),60,1)
    rotate_60=cv2.warpAffine(img,M,(cols,rows))
    X_augment=np.concatenate((X_augment,rotate_60[np.newaxis,:]),axis=0)
    y_augment=np.append(y_augment,label)
         
    #90度旋转
    rows,cols=img.shape[:2]
    M=cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    rotate_90=cv2.warpAffine(img,M,(cols,rows))
    X_augment=np.concatenate((X_augment,rotate_90[np.newaxis,:]),axis=0)
    y_augment=np.append(y_augment,label)

print("X_augment.shape: ", X_augment.shape)
print("y_augment.shape: ", y_augment.shape)



X_train = X_augment
y_train = y_augment

# shuffle
train_permutation = np.random.permutation(X_train.shape[0])
test_permutation = np.random.permutation(X_test.shape[0])
X_train,y_train, = X_train[train_permutation,:,:,:], y_train[train_permutation] 
X_test,y_test = X_test[test_permutation,:,:,:], y_test[test_permutation]

print("Total number of train : ", len(y_train))
print("train_class_1 num : ", y_train.tolist().count(0))
print("train_class_2 num : ", y_train.tolist().count(1))
print("train_class_3 num : ", y_train.tolist().count(2))
print("train_class_4 num : ", y_train.tolist().count(3))
print("")
print("Total number of test : ", len(y_test))
print("test_class_1 num : ", y_test.tolist().count(0))
print("test_class_2 num : ", y_test.tolist().count(1))
print("test_class_3 num : ", y_test.tolist().count(2))
print("test_class_4 num : ", y_test.tolist().count(3))


# データ前処理

class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        data = x_train.astype('float32')
        self.x_train = []
        for i in range(data.shape[0]):
            self.x_train.append(Image.fromarray(np.uint8(data[i])))
        self.y_train = y_train
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.transform(self.x_train[idx]), torch.tensor(y_train[idx], dtype=torch.long)

trainval_data = train_dataset(X_train, y_train)
test_data = train_dataset(X_test, y_test)


batch_size = 32
val_size = int(len(trainval_data)*0.2)
train_size = len(trainval_data) - val_size

train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

dataloader_valid = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True
)


print("##########################")
print("CNN train :  ")

rng = np.random.RandomState(1234)
random_state = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
conv_net = nn.Sequential(
    nn.Conv2d(3, 32, 5),              # Conv 512x512x3 -> 508x508x32
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.AvgPool2d(2),                  # Pool 508x508x32 -> 254x254x32
    
    nn.Conv2d(32, 32, 5),             # Conv 254x254x32-> 250x250x32
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.AvgPool2d(2),                  # Pool 250x250x32 -> 125x125x32

    nn.Conv2d(32, 64, 3),            # Conv 125x125x32 -> 123x123x64
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.AvgPool2d(2),                  # Pool 123x123x64 -> 61x61x64
    
    nn.Conv2d(64, 64, 3),            # Conv 61x61x64 -> 59x59x64
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.AvgPool2d(2),                  # Pool 59x59x64 -> 29x29x64

    nn.Conv2d(64, 128, 3),           # Conv 29x29x64-> 27x27x128
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.AvgPool2d(2),                  # Pool 27x27x128 -> 13x13x128

    nn.Conv2d(128, 128, 3),           # Conv 13x13x128 -> 11x11x128
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.AvgPool2d(2),                  # Pool 11x11x128-> 5x5x128
    
    nn.Flatten(),
    
    nn.Linear(5*5*128, 1024),
    torch.nn.Dropout(0.5),
    nn.ReLU(),
    
    nn.Linear(1024, 256),
    torch.nn.Dropout(0.5),
    nn.ReLU(),
    
    nn.Linear(256, 4)
)

losstrain=[]
lossvalid=[]
Accuracytrain=[]
Accuracyvalid=[]


device = 'cuda'
conv_net.to(device)


# TRAIN 
n_epochs = 100
lr = 0.01
optimizer = optim.SGD(conv_net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()  

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    # Train
    conv_net.train()
    n_train = 0
    acc_train = 0
    for x, t in dataloader_train:
        n_train += t.size()[0]
        conv_net.zero_grad()  # 勾配の初期化
        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)
        y = conv_net.forward(x)  # 順伝播
        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算
        loss.backward()  # 誤差の逆伝播
        optimizer.step()  # パラメータの更新
        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする
        acc_train += (pred == t).float().sum().item()
        losses_train.append(loss.tolist())
    # Evaluate
    conv_net.eval()
    n_val = 0
    acc_val = 0
    for x, t in dataloader_valid:
        n_val += t.size()[0]
        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)
        y = conv_net.forward(x)  # 順伝播
        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算
        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする
        acc_val += (pred == t).float().sum().item()
        losses_valid.append(loss.tolist())
    # Visualize loss & accuracy    
    losstrain.append(np.mean(losses_train))   
    Accuracytrain.append(acc_train/n_train)
    lossvalid.append(np.mean(losses_train))
    Accuracyvalid.append(acc_val/n_val)
    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(epoch,np.mean(losses_train),acc_train/n_train,np.mean(losses_valid),acc_val/n_val))


# train processing plot
epochs=range(1,n_epochs+1)
plt.figure()
plt.plot(epochs,Accuracytrain,'b',label='Training accuracy')  
plt.plot(epochs, Accuracyvalid,'r',label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.savefig('convnet_train.jpg')
print("the picture has been saved")
print(">[*_*]<")



# TEST
print("CNN test :  ")
losses_test = []
n_test = 0
acc_test = 0
conv_net.eval()

for x, t in dataloader_test:
        n_test += t.size()[0]
        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)
        y = conv_net.forward(x)  # 順伝播
        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算
        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする
        acc_test += (pred == t).float().sum().item()
        losses_test.append(loss.tolist())

# Visualize loss & accuracy    
print('Loss: {:.3f}, Accuracy: {:.3f}'.format(np.mean(losses_test),acc_test/n_test))


# ResNet18遷移学習

# FC層のクラス数を変更
print("######################")
print("ResNet train :  ")
model = models.resnet18(pretrained=True)
num_fcs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_fcs, 256),
    nn.Dropout(p=0.5),
    nn.ReLU(inplace=True),
    nn.Linear(256, 4)
)
model.avgpool = nn.AdaptiveAvgPool2d(1)
model = model.cuda() 

# TRAIN
n_epochs = 100
lr = 0.01
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
device = 'cuda'

losstrain=[]
lossvalid=[]
Accuracytrain=[]
Accuracyvalid=[]

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    # Train
    model.train()
    n_train = 0
    acc_train = 0
    for x, t in dataloader_train:
        n_train += t.size()[0]
        model.zero_grad()  # 勾配の初期化
        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)
        y = model.forward(x)  # 順伝播
        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算
        loss.backward()  # 誤差の逆伝播
        optimizer.step()  # パラメータの更新
        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする
        acc_train += (pred == t).float().sum().item()
        losses_train.append(loss.tolist())
    # Evaluate
    model.eval()
    n_val = 0
    acc_val = 0
    for x, t in dataloader_valid:
        n_val += t.size()[0]
        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)
        y = model.forward(x)  # 順伝播
        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算
        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする
        acc_val += (pred == t).float().sum().item()
        losses_valid.append(loss.tolist())
    # Visualize loss & accuracy    
    losstrain.append(np.mean(losses_train))   
    Accuracytrain.append(acc_train/n_train)
    lossvalid.append(np.mean(losses_train))
    Accuracyvalid.append(acc_val/n_val)
    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(epoch,np.mean(losses_train),acc_train/n_train,np.mean(losses_valid),acc_val/n_val))


# train processing plot
epochs=range(1,n_epochs+1)
plt.plot(epochs,Accuracytrain,'b',label='Training accuracy')  
plt.plot(epochs, Accuracyvalid,'r',label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.savefig('convnet_train.jpg')
print("the picture has been saved")
print(">[*_*]<")


# TEST
print("ResNet test :  ")
losses_test = []
n_test = 0
acc_test = 0
conv_net.eval()

for x, t in dataloader_test:
        n_test += t.size()[0]
        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)
        y = model.forward(x)  # 順伝播
        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算
        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする
        acc_test += (pred == t).float().sum().item()
        losses_test.append(loss.tolist())

# Visualize loss & accuracy    
print('Loss: {:.3f}, Accuracy: {:.3f}'.format(np.mean(losses_test),acc_test/n_test))









