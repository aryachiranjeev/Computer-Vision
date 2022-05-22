import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
!pip install opencv-python==3.4.2.17
!pip install opencv-contrib-python==3.4.2.17


import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

check = True
for i in range(1, 6):
  b = unpickle("/DATA1/Chiranjeev/CV_Assignment/Cifar-10/data_batch_"+str(i))
  if check:
    bimages = b[b'data']
    labels = np.array(b[b'labels'])
    check = False
    continue
  bimages = np.concatenate((bimages, b[b'data']))
  labels = np.concatenate((labels, np.array(b[b'labels'])))

images = []
for i in range(bimages.shape[0]):
  images.append(bimages[i].reshape(-1).reshape(3, 32, 32).transpose(1, 2, 0))

trainimages = np.array(images)
trainlabels = labels


b = unpickle("/DATA1/Chiranjeev/CV_Assignment/Cifar-10/test_batch")
if check:
  bimages = b[b'data']
  labels = np.array(b[b'labels'])

images = []
for i in range(bimages.shape[0]):
  images.append(bimages[i].reshape(-1).reshape(3, 32, 32).transpose(1, 2, 0))

testimages = np.array(images)
testlabels = labels

def generateHogFeatures(images):
  hog_features = []
  hog_images = []
  for j, i in enumerate(images):
    fd, hog_image = hog(i, orientations=8, pixels_per_cell=(4, 4),
                      cells_per_block=(1, 1), visualize=True, multichannel = True)
    hog_images.append(hog_image)
    hog_features.append(fd.reshape(-1))
  hog_features = np.array(hog_features)
  return hog_features, hog_images


def generateSiftFeatures(images):
  sift_features = []
  sift_images = []
  sift = cv2.xfeatures2d.SIFT_create()
  for j, i in enumerate(images):
    sift_features.append(np.array(sift.detectAndCompute(i, None)[1]))
    img = cv2.drawKeypoints(i, sift.detectAndCompute(i, None)[0], None, color=(255, 255, 0))
    sift_images.append(img)
  sift_features = np.array(sift_features)
  return sift_features, sift_images

hf, hi = generateHogFeatures(trainimages)
sf, si = generateSiftFeatures(trainimages)

t_hf = hf.reshape(-1, 1024)
model = MLPClassifier(solver='adam', hidden_layer_sizes=(20,40), max_iter=100)
model.fit(t_hf, trainlabels)

print(accuracy_score(testlabels, model.predict(testimages)))


# https://www.geeksforgeeks.org/ml-getting-started-with-alexnet/
model = Sequential()
  
# 1st Convolutional Layer
model.add(Conv2D(filters = 96, input_shape = (224, 224, 3), 
            kernel_size = (11, 11), strides = (4, 4), 
            padding = 'valid'))
model.add(Activation('relu'))
# Max-Pooling 
model.add(MaxPooling2D(pool_size = (2, 2),
            strides = (2, 2), padding = 'valid'))
# Batch Normalisation
model.add(BatchNormalization())
  
# 2nd Convolutional Layer
model.add(Conv2D(filters = 256, kernel_size = (11, 11), 
            strides = (1, 1), padding = 'valid'))
model.add(Activation('relu'))
# Max-Pooling
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), 
            padding = 'valid'))
# Batch Normalisation
model.add(BatchNormalization())
  
# 3rd Convolutional Layer
model.add(Conv2D(filters = 384, kernel_size = (3, 3), 
            strides = (1, 1), padding = 'valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())
  
# 4th Convolutional Layer
model.add(Conv2D(filters = 384, kernel_size = (3, 3), 
            strides = (1, 1), padding = 'valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())
  
# 5th Convolutional Layer
model.add(Conv2D(filters = 256, kernel_size = (3, 3), 
            strides = (1, 1), padding = 'valid'))
model.add(Activation('relu'))
# Max-Pooling
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), 
            padding = 'valid'))
# Batch Normalisation
model.add(BatchNormalization())
  
# Flattening
model.add(Flatten())
  
# 1st Dense Layer
model.add(Dense(4096, input_shape = (224*224*3, )))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())
  
# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Softmax Layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

features = []
for i in h_tf:
  features.append(model.predict(i))

t_hf = hf.reshape(-1, 1024)
model = MLPClassifier(solver='adam', hidden_layer_sizes=(20,40), max_iter=100)
model.fit(features, trainlabels)

print(accuracy_score(testlabels, model.predict(testimages)))



# Read lFW Dataset
test = open("/content/drive/MyDrive/LFW/test.txt").read().split("\n")
data = []
for i in range(1, int(test[0])+1):
  items = test[i].split("\t")
  temp = []
  index = "0000000"+items[1]
  temp.append(items[0]+"/"+items[0]+"_"+index[-4:]+".jpg")
  index = "0000000"+items[2]
  temp.append(items[0]+"/"+items[0]+"_"+index[-4:]+".jpg")
  temp.append(1.0)
  data.append(temp)

for i in range(int(test[0])+1, len(test)):
  items = test[i].split("\t")
  temp = []
  index = "0000000"+items[1]
  temp.append(items[0]+"/"+items[0]+"_"+index[-4:]+".jpg")
  index = "0000000"+items[3]
  temp.append(items[2]+"/"+items[2]+"_"+index[-4:]+".jpg")
  temp.append(0.0)
  data.append(temp)


def generateLBP(img):
  lbpImage = np.zeros((img.shape[0], img.shape[1]))
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  for i in range(1, gray.shape[0]-1):
    for j in range(1, gray.shape[1]-1):
      lbp = []
      for l in range(1,-2, -1):
        if gray[i-1][j-l] > gray[i][j]:
          lbp.append(1)
        else:
          lbp.append(0)
      
      if gray[i][j+1] > gray[i][j]:
        lbp.append(1)
      else:
        lbp.append(0)

      for l in range(1,-2, -1):
        if gray[i+1][j+l] > gray[i][j]:
          lbp.append(1)
        else:
          lbp.append(0)
      
      if gray[i][j-1] > gray[i][j]:
        lbp.append(1)
      else:
        lbp.append(0)
      val = 0
      for l, k in enumerate(lbp):
        val+= pow(2, l) * k
      lbpImage[i-1][j-1] = val
  return lbpImage[:-2, :-2].flatten()


from scipy import spatial
from google.colab.patches import cv2_imshow
import cv2

scores = []
labels = []

for i in range(len(data)):
  try:
    img1 = cv2.cvtColor(cv2.imread("/content/drive/MyDrive/LFW/LFWDataset/lfw/"+data[i][0]), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("/content/drive/MyDrive/LFW/LFWDataset/lfw/"+data[i][1]), cv2.COLOR_BGR2RGB)
    f1 = generateLBP(img1)
    f2 = generateLBP(img2)
    scores.append(1 - spatial.distance.cosine(f1, f2))
    labels.append(data[i][2])
  except:
    continue
# print(scores)
# print(labels)



# https://github.com/AlfredXiangWu/LightCNN/blob/master/light_cnn.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

class network_9layers(nn.Module):
    def __init__(self, num_classes=79077):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out, x

class network_29layers(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers, self).__init__()
        self.conv1  = mfm(1, 48, 5, 1, 2)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc     = mfm(8*8*128, 256, type=0)
        self.fc2    = nn.Linear(256, num_classes)
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)
        return out, fc


class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers_v2, self).__init__()
        self.conv1    = mfm(1, 48, 5, 1, 2)
        self.block1   = self._make_layer(block, layers[0], 48, 48)
        self.group1   = group(48, 96, 3, 1, 1)
        self.block2   = self._make_layer(block, layers[1], 96, 96)
        self.group2   = group(96, 192, 3, 1, 1)
        self.block3   = self._make_layer(block, layers[2], 192, 192)
        self.group3   = group(192, 128, 3, 1, 1)
        self.block4   = self._make_layer(block, layers[3], 128, 128)
        self.group4   = group(128, 128, 3, 1, 1)
        self.fc       = nn.Linear(8*8*128, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=False)
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        x = F.dropout(fc, training=self.training)
        out = self.fc2(x)
        return out, fc

def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs)
    return model

def LightCNN_29Layers(**kwargs):
    model = network_29layers(resblock, [1, 2, 3, 4], **kwargs)
    return model

def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model

model.load_state_dict(torch.load("https://drive.google.com/file/d/0ByNaVHFekDPRMGlLWVBhbkVGVm8/view"))

from scipy import spatial
from google.colab.patches import cv2_imshow
import cv2
import torchvision.transforms as T

scores = []

transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128,128)),
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5))
        ])

label_new = []
ctr = 0
for i in range(len(data)):
  img1 = cv2.imread("/content/drive/MyDrive/LFW/LFWDataset/lfw/"+data[i][0],0)
  img2 = cv2.imread("/content/drive/MyDrive/LFW/LFWDataset/lfw/"+data[i][1],0)

  print(i)
  img1 = transform(img1).unsqueeze(0).float().to(device)
  img2 = transform(img2).unsqueeze(0).float().to(device)
  label_new.append(data[i][2])
  
  # except:
  #   ctr+=1
  #   continue
  
  f1 = model(img1)
  f2 = model(img2)
  scores.append(1 - spatial.distance.cosine(f1.detach().cpu().numpy(), f2.detach().cpu().numpy()))

from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve( labels, scores, pos_label = 1)

plt.xscale('log')
plt.title("ROC Curve for LBP Verification")
plt.plot(fpr, tpr, color = 'r')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("/content/drive/MyDrive/LFW/LFWDataset/LBPPlot2.png")
plt.show()