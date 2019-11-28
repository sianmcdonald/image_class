# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# imports
import torch
from torchvision import datasets, transforms, models
import json
import torch.nn.functional as F
from torch import nn, optim
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse

#('-l','--list', action='append', help='<Required> Set flag', required=True)
#('-l','--list', nargs='+', help='<Required> Set flag', required=True)
#Argparse step
parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--data_dir', type=str, help='Input dir for data')
parser.add_argument('--save_dir', type=str, help='Output dir for data')
parser.add_argument('--arch', type=str, default = "vgg16" ,help='Architecture')
parser.add_argument('--learning_rate', type=float, default = 0.001, help='The learning rate for your algo')
parser.add_argument('-l','--hidden_unit', nargs='+', type=int, default = 4096 , help='Hidden unit - provide 2')
parser.add_argument('--epochs', type=int, default = 2, help='The number of epochs for algo')
parser.add_argument('--gpu', type=str, default = "cuda", help='Use GPU or not')
parser.add_argument('--batch_size', type=int,default = 64, help='Batch size for loading data')
args = parser.parse_args()
#print(args.hidden_unit[1])
#print(args.hidden_unit[0])

#Loading in the data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    val_data = datasets.ImageFolder(valid_dir, transform = test_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size)
    validloader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size)

    return trainloader, testloader, validloader

trainloader, testloader, validloader = load_data(args.data_dir)

#print(trainloader)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Importing the model
model = models.vgg16(pretrained=True)


for param in model.parameters():
    param.requires_grad = False
        
classifier = nn.Sequential(nn.Linear(25088,args.hidden_unit[0]),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(args.hidden_unit[0],args.hidden_unit[1]),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(args.hidden_unit[1],102),
                               nn.LogSoftmax(dim=1))
    
model.classiifier = classifier

#Define criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)


#Put model on GPU
model.to(args.gpu)

#print(model)

#Training the model

def training(trainloader = trainloader,optimizer = optimizer,model = model, criterion=criterion, validloader=validloader):
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        model.train()
        for inputs, labels in iter(trainloader):
            steps += 1
            inputs,labels = inputs.to(args.gpu), labels.to(args.gpu)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            
            if steps % print_every ==0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in iter(validloader):
                        
                        inputs, labels = inputs.to(args.gpu), labels.to(args.gpu)
                        
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                       f"Train loss: {running_loss/print_every:.3f}.. "
                       f"Test loss: {test_loss/len(validloader):.3f}.. "
                       f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train() 

    ########################################

training()
        



  
       


