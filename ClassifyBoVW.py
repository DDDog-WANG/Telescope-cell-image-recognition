import cv2,os
import numpy as np
from matplotlib import pyplot as plt
import sys


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
        DataLabel.append(label[0])
print(len(DataPath))
print(len(DataLabel))

# 特徴点の抽出
k = 10
detector = cv2.SIFT_create()
trainer = cv2.BOWKMeansTrainer(k)

keypoints = []
directors = []

DataPathremove = []
DataLabelremove = []

for i in range(len(DataPath)):
    path = DataPath[i]
    img = cv2.imread(path)
    ks, ds = detector.detectAndCompute(img, None)
    if len(ks) == 0:
        print("Keypoint of ", path," is None, so remove it from DataPath")
    else:
        DataPathremove.append(DataPath[i])
        DataLabelremove.append(DataLabel[i])
        trainer.add(ds.astype(np.float32))
        keypoints.append(ks)
        directors.append(ds)
     
DataPath = DataPathremove
DataLabel = DataLabelremove
print("After remove it, len(DataPath) is ", len(DataPath))

# クラスタリング
dictionary = trainer.cluster()
print("dictionary: ",dictionary.shape)

# ヒストグラム
matcher = cv2.BFMatcher()
extractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
extractor.setVocabulary(dictionary)
descriptor_list = []
for path in DataPath:
    img = cv2.imread(path)
    ks = detector.detect(img, None)
    descriptor = extractor.compute(img, ks)[0]
    descriptor_list.append(descriptor)

print("descriptor: ",type(descriptor))
print("descriptor: ",descriptor.shape)
print("descriptor_list: ",type(descriptor_list))
print("descriptor_list: ",len(descriptor_list))


# BPNN画像分類
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import regularizers

# divide each classes into train and test 
data_descriptor = np.array(descriptor_list)
data_intlabel = LabelEncoder().fit(DataLabel).transform(DataLabel)

data_descriptor_class=[[],[],[],[]]
data_intlabel_class=[[],[],[],[]]
for i in range(len(data_intlabel)):
    data_descriptor_class[data_intlabel[i]].append(data_descriptor[i])
    data_intlabel_class[data_intlabel[i]].append(data_intlabel[i])

X_train, X_test, y_train, y_test = [],[],[],[]
for i in range(len(data_descriptor_class)):
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(np.array(data_descriptor_class[i]), np.array(data_intlabel_class[i]), test_size=0.3, random_state=42)        
    X_train.extend(X_train_class), X_test.extend(X_test_class), y_train.extend(y_train_class), y_test.extend(y_test_class)

X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# shuffle
train_permutation = np.random.permutation(X_train.shape[0])
test_permutation = np.random.permutation(X_test.shape[0])
X_train,y_train, = X_train[train_permutation, :],y_train[train_permutation] 
X_test,y_test = X_test[test_permutation, :],y_test[test_permutation]

# Model
model_l2_dropout = keras.Sequential([
    keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, kernel_regularizer=regularizers.l2(0.001),activation='softmax')
])

model_l2_dropout.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
epochs = 100
model_l2_dropout.fit(X_train, y_train, epochs=epochs)

# Test
test_loss, test_acc = model_l2_dropout.evaluate(X_test, y_test)
print('\n model_l2_dropout Test accuracy:', test_acc)



