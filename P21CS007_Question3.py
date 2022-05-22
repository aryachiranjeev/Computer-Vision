import numpy as np 
import pandas as pd 
import sys
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from torchvision.utils import save_image


device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
batch_size = 64
epochs = 15
lr = 0.001
criterion = nn.CrossEntropyLoss()
test_batch_size = 8

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        
        layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*layers[:5]) # size=(N, 64, x.H/2, x.W/2)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.layer3 = layers[6]  # size=(N, 256, x.H/8, x.W/8)
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.layer4 = layers[7]  # size=(N, 512, x.H/16, x.W/16)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        
        self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, n_class, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)
        
        merge = torch.cat([up1, up2, up3, up4], dim=1)
        merge = self.conv1k(merge)
        out = self.sigmoid(merge)
        
        return out

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        # Define the layers here
        # Note: for conv, use a padding of (1,1) so that size is maintained
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,padding = 1)
        self.bn = nn.BatchNorm2d(out_ch,momentum = 0.1)
        self.relu = nn.ReLU()
    def forward(self, x):
        # define forward operation using the layers above
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class down_layer(nn.Module):
    def __init__(self):
        super(down_layer, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # use nn.MaxPool2d( )        
    def forward(self, x):
        x1,idx = self.down(x)
        return x1,idx

class un_pool(nn.Module):
    def __init__(self):
        super(un_pool, self).__init__()       
        self.un_pool = nn.MaxUnpool2d(kernel_size=2, stride=2) # use nn.Upsample() with mode bilinear
        
    
    def forward(self, x, idx,x1):
        #Take the indicies from maxpool layer
        x = self.un_pool(x,idx,output_size = x1.size())
        return x 

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # 1 conv layer
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,padding = 1)

    def forward(self, x):
        # Forward conv layer
        x = self.conv(x)
        return x

class SegNet(nn.Module):
    def __init__(self, n_channels_in, n_classes):
        super(SegNet, self).__init__()
        self.conv1 = single_conv(n_channels_in,64)
        self.conv2 = single_conv(64,64)
        self.down1 = down_layer()
        self.conv3 = single_conv(64,128)
        self.conv4 = single_conv(128,128)
        self.down2 = down_layer()
        self.conv5 = single_conv(128,256)
        self.conv6 = single_conv(256,256)
        self.conv7 = single_conv(256,256)
        self.down3 = down_layer()
        self.conv8 = single_conv(256,512)
        self.conv9 = single_conv(512,512)
        self.conv10 = single_conv(512,512)
        self.down4 = down_layer()
        self.conv11 = single_conv(512,512)
        self.conv12 = single_conv(512,512)
        self.conv13 = single_conv(512,512)
        self.down5 = down_layer()
        self.up1 = un_pool()
        self.conv14 = single_conv(512,512)
        self.conv15 = single_conv(512,512)
        self.conv16 = single_conv(512,512)
        self.up2 = un_pool()
        self.conv17 = single_conv(512,512)
        self.conv18 = single_conv(512,512)
        self.conv19 = single_conv(512,256)
        self.up3 = un_pool()
        self.conv20 = single_conv(256,256)
        self.conv21 = single_conv(256,256)
        self.conv22 = single_conv(256,128)
        self.up4 = un_pool()
        self.conv23 = single_conv(128,128)
        self.conv24 = single_conv(128,64)
        self.up5 = un_pool()
        self.conv25 = single_conv(64,64)
        self.outconv1 = outconv(64,n_classes)

    def forward(self, x):
        # Define forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3,idx1 = self.down1(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6,idx2 = self.down2(x5)
        x7 = self.conv5(x6)
        x8 = self.conv6(x7)
        x9 = self.conv7(x8)
        x10,idx3 = self.down3(x9)
        x11 = self.conv8(x10)
        x12 = self.conv9(x11)
        x13 = self.conv10(x12)
        x14,idx4 = self.down4(x13)
        x15 = self.conv11(x14)
        x16 = self.conv12(x15)
        x17 = self.conv13(x16)
        x18,idx5 = self.down5(x17)
        x19 = self.up1(x18,idx5,x17)
        x20 = self.conv14(x19)
        x21 = self.conv15(x20)
        x22 = self.conv16(x21)
        x23 = self.up2(x22,idx4,x13)
        x24 = self.conv17(x23)
        x25 = self.conv18(x24)
        x26 = self.conv19(x25)
        x27 = self.up3(x26,idx3,x9)
        x28 = self.conv20(x27)
        x29 = self.conv21(x28)
        x30 = self.conv22(x29)
        x31 = self.up4(x30,idx2,x5)
        x32 = self.conv23(x31)
        x33 = self.conv24(x32)
        x34 = self.up4(x33,idx1,x2)
        x35 = self.conv25(x34)
        x = self.outconv1(x35)
        return x

class CityscapeDataset(Dataset):
    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.label_model = label_model
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_fn = self.images[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        cityscape, label = self.split_image(image)
        # label = self.transform(label)
        # label_class = self.label_model.predict(label.reshape(-1, 3).resize(224, 224))
        label_class = self.label_model.predict(cv2.resize(label,(224,224)).reshape(-1, 3))
        # print("Label Class Before: ", label_class.shape)
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label_class.reshape(224,224)).long()
        # print("Label Class After: ", label_class.shape)
        return cityscape, label_class
    
    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label
    
    def transform(self, image):
        transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transformation(image)

def labelling_model():
    num_items = 1000
    color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
    num_classes = 19
    label_model = KMeans(n_clusters=num_classes)
    label_model.fit(color_array)
    return label_model
import skimage.io as io
import cv2
def train(data_loader, model):
    step_losses = []
    epoch_losses = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
            # print(Y[0])
            Y=Y.float()
            Y1 = Y[0].detach().cpu().numpy()
            # print(np.unique(Y1))
            # cv2.imwrite( "hojaa_save.png",Y1.astype(np.uint8)*255.0)
            # io.imsave("test.jpg",Y1)
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_pred = model(X)
            # print("Y_pred: ", Y_pred.shape)
            # print("Y: ", Y.shape)
            loss = criterion(Y_pred, Y.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
        epoch_losses.append(epoch_loss/len(data_loader))
    return step_losses, epoch_losses, model

def plots(step_losses, epoch_losses):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[0].set_title("Step Losses")
    axes[1].plot(epoch_losses)
    axes[1].set_title("Epoch Losses")
    plt.savefig("./Qs3_Output/" + str(sys.argv[1]) + "/losses.png")

def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


def test(data_loader, model_test):
    X, Y = next(iter(data_loader))
    X, Y = X.to(device), Y.to(device)
    Y_pred = model_test(X)
    Y_pred = torch.argmax(Y_pred, dim=1)


    inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])

    fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))

    iou_scores = []
    dice_scores = []
    for i in range(test_batch_size):
        
        landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
        label_class = Y[i].cpu().detach().numpy()
        label_class_predicted = Y_pred[i].cpu().detach().numpy()

        #Dice
        dice_score = dice(label_class_predicted, label_class)
        dice_scores.append(dice_score)
        
        # IOU score
        intersection = np.logical_and(label_class, label_class_predicted)
        union = np.logical_or(label_class, label_class_predicted)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)

        axes[i, 0].imshow(landscape)
        axes[i, 0].set_title("Landscape")
        axes[i, 1].imshow(label_class)
        axes[i, 1].set_title("Label Class")
        axes[i, 2].imshow(label_class_predicted)
        axes[i, 2].set_title("Label Class - Predicted")
    plt.savefig("./Qs3_Output/" + str(sys.argv[1]) + "/prediction.png")
    print("IOU: ", sum(iou_scores) / len(iou_scores))
    print("DICE: ", sum(iou_scores) / len(iou_scores))

def main():
    data_dir = "./data/cityscapes_data"
    train_dir = os.path.join(data_dir, "train") 
    val_dir = os.path.join(data_dir, "val")
    train_imgs = os.listdir(train_dir)
    val_imgs = os.listdir(val_dir)
    label_model = labelling_model()
    print("Train size:", len(train_imgs))
    print("Validation size:", len(val_imgs))
    print("Device: ", device)

    #Training
    dataset = CityscapeDataset(train_dir, label_model)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    if (str(sys.argv[1]) == "fcn"):
        model = FCN(19).to(device)
    else:
        model = SegNet(3,19).to(device)
    step_losses, epoch_losses, model = train(data_loader, model)
    plots(step_losses, epoch_losses)
    torch.save(model.state_dict(), "./Qs3_Output/" + str(sys.argv[1]) + "/" + str(sys.argv[1]) + ".pth")

    #Testing
    if (str(sys.argv[1]) == "fcn"):
        model = FCN(19).to(device)
    else:
        model = SegNet(3,19).to(device)
    model_test.load_state_dict(torch.load("./Qs3_Output/" + str(sys.argv[1]) + "/" + str(sys.argv[1]) + ".pth"))
    dataset = CityscapeDataset(val_dir, label_model)
    data_loader = DataLoader(dataset, batch_size=test_batch_size)
    test(data_loader, model_test)

if __name__ == "__main__":
    main()