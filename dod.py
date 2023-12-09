import os
import random
import numpy as np
import pandas as pd
import time
# from tqdm import tqdm
import torch
print(torch.__version__)
print(pd.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

from facenet_pytorch import MTCNN

# детектор - предобучен
IMAGE_SIZE = [224, 224]
detector = MTCNN(image_size=IMAGE_SIZE, margin=0, min_face_size=40, thresholds=[0.8, 0.7, 0.8])

# преобразователи картинки
transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])

transform_test=transforms.Compose([
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])

# датасет для обучения
dataset0=datasets.ImageFolder(root="gender/train",transform=None)
# имена классов
class_names=dataset0.classes
print(class_names)
print(len(class_names))

s = input('готовы учить (д/н):')

# чтение данных
class DataModule(pl.LightningDataModule):
    
    def __init__(self, transform=transform, batch_size=32):
        super().__init__()
        self.train_dir = "gender/train"
        self.val_dir = "gender/valid"
        self.test_dir = "gender/test"
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        trainset = datasets.ImageFolder(root=self.train_dir, transform=self.transform)
        valset = datasets.ImageFolder(root=self.val_dir, transform=self.transform)
        testset = datasets.ImageFolder(root=self.test_dir, transform=self.transform)

        self.train_dataset = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.val_dataset = DataLoader(valset, batch_size=self.batch_size)
        self.test_dataset = DataLoader(testset, batch_size=self.batch_size)

    def train_dataloader(self):
        return self.train_dataset
    
    def val_dataloader(self):
        return self.val_dataset
    
    def test_dataloader(self):
        return self.test_dataset

#  нейронка
class ConvolutionalNetwork(LightningModule):
    
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1) 
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, len(class_names))

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)    

# Обучение сети        
if s=='д':
    
    datamodule = DataModule()
    datamodule.setup()
    model = ConvolutionalNetwork()
    trainer = pl.Trainer(max_epochs=15) # создание обучалки
    trainer.fit(model, datamodule) # обучаем
    
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader() #  тестируем
    trainer.test(dataloaders=test_loader)    

    

    # Сохранить в файл
    m = torch.jit.script(model)
    torch.jit.save(m, 'scriptmodule.pt')


s = input('готовы работать (д/н): ')    

if s == 'д': 

        # Load all tensors to the original device
        model = torch.jit.load('scriptmodule.pt')
        print('model: ' + 'scriptmodule.pt')

        device = torch.device("cpu")   #"cuda:0"

        model.eval() # исполнение модели
        # для вывода результата
        import cv2 

        font = cv2.FONT_HERSHEY_SIMPLEX 
         
        fontScale = 1
        color = (255, 0, 0) 

        thickness = 2

        vid = cv2.VideoCapture(0) 

        while(True): 

            # Capture the video frame 
            # by frame 
            ret, frame = vid.read() 
            if ret:
                image  = Image.fromarray(frame)
                
                try:
                    print(image.size)
                    boxes, probs, landmarks = detector.detect(image, landmarks=True) # детектор
                    print('detect', boxes)
                    e = ''
                    for box in boxes:
                        x1, y1, x2, y2 = np.array(boxes[0]).astype(int)
                        img = transform_test(image.crop([x1,y1,x2,y2]))
                        yp = model(img)# работа модели
                        pred = (yp[0,0]>yp.mean()*0.5).int() # получаем решение
                        print(yp[0, :], pred)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
                        org = (x1, y1)
                        frame = cv2.putText(frame, str(class_names[pred]), org, font, fontScale, color, thickness, cv2.LINE_AA)
                        
                        print(pred)
                        e += str(class_names[pred]) + ' : '
                except:
                    e = 'no Face'
                    print('No face')
                
                try:    
                    x1, y1, x2, y2 = np.array(boxes[0]).astype(int)
                except:
                    print(image.size)
                time.sleep(0.1)
                # Display frame 
                cv2.imshow('frame :'+ e, frame) 

                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

        # After the loop release the cap object 
        vid.release() 
        # Destroy all the windows 
        cv2.destroyAllWindows() 