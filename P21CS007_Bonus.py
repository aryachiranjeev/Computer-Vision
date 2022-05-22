import cv2
import json
import pandas as pd
import skimage.io as io
import numpy as np
import os


f = open('entries.json')
d = json.load(f)


mask_labels = []
imgs = []
ctr=0
for k in d.keys():
    path1 = "/DATA1/Chiranjeev/CV_Assignment_3/Q4/images/"+k
    if os.path.exists(path1)==True:
        ctr+=1
        img = cv2.imread(path1,0)
        a = d[k]
        mask = np.zeros_like(img)
        for dic in a:
            bbox = dic['bbox']
            x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            if dic['category_id'] == "text":
                mask[y1:y2,x1:x2]=1
            elif dic['category_id'] == "title":
                mask[y1:y2,x1:x2]=2
            elif dic['category_id'] == "figure":
                mask[y1:y2,x1:x2]=3
            elif dic['category_id'] == "graph":
                mask[y1:y2,x1:x2]=4
            elif dic['category_id'] == "table":
                mask[y1:y2,x1:x2]=5
        img = cv2.resize(img,(256,256))
        mask = cv2.resize(mask,(256,256))
        mask_labels.append(mask)
        imgs.append(img)
#         print(ctr)
    elif os.path.exists(path1)==False:
        continue
        
        
from simple_multi_unet_model import multi_unet_model
import tensorflow as tf
import tensorflow
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np


train_images = np.asarray(imgs)
train_masks = np.asarray(mask_labels)

SIZE_X = 512 
SIZE_Y = 512
n_classes=6 

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
print(train_masks.shape)
print(train_images.shape)
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)


train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  

from tensorflow.keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_masks_reshaped_encoded),
                                                 train_masks_reshaped_encoded)
print("Class weights are...:", class_weights)


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train_cat, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=10, 
                    validation_data=(X_test, y_test_cat), 
                    class_weight=class_weights,
                    shuffle=False)
                    


_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

acc = history.history['acc']
val_acc = history.history['val_acc']

y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU
n_classes = 6
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)


import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

cv2.imwrite("predicted_img_1.jpg",predicted_img)
cv2.imwrite("predicted_img_2.jpg",predicted_img)
cv2.imwrite("predicted_img_3.jpg",predicted_img)

