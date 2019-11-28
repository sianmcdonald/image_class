#Imports
import torch
from torchvision import datasets, transforms, models
import json
import torch.nn.functional as F
from torch import nn, optim
from collections import OrderedDict
import numpy as np
import argparse

#Argparse step
parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--data_dir', type=str, default = 'flowers', help='Input directory for data')
parser.add_argument('--save_dir', type=str, help='Output directory for data')
parser.add_argument('--arch', type=str, default = "vgg16", help='Model architecture - pick either vgg16 or resnet101')
parser.add_argument('--learning_rate', type=float, default = 0.001, help='The learning rate for your algorithm')
parser.add_argument('-l','--hidden_unit', nargs='+', type=int, default = 1020 , help='Hidden units - provide 2 integers')
parser.add_argument('--epochs', type=int, default = 4, help='The number of epochs for algorithm')
parser.add_argument('--gpu', type=str, default = "cuda", help='Use GPU or not - input cude or cpu')
parser.add_argument('--batch_size', type=int,default = 64, help='Batch size for loading data')
args = parser.parse_args()

#Loading data function
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

    return trainloader, testloader, validloader, train_data


#Training function
def training(trainloader ,optimizer ,model , criterion, validloader):
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

#Fucntion that import model, creates the new classifier and defines criterion and optimizer
def model_stuff(architecture = args.arch, hidden_unit = args.hidden_unit, lr = args.learning_rate):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
    elif architecture == 'resnet101':
        model = models.resnet101(pretrained=True)
        in_features = 2048
    else:
        print('Please choose between vgg16 or resnet101')
        
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(nn.Linear(in_features,args.hidden_unit[0]),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(args.hidden_unit[0],args.hidden_unit[1]),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(args.hidden_unit[1],102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    model.to(args.gpu)
    
    return cat_to_name, model,criterion,optimizer


#Saving model
def saving(arch, model, epochs,train_data):
    checkpoint = {'Architecture': args.arch,
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'class_to_idx':train_data.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
  
  

#Part 2 - predict
 
#Loading model
def load_checkpoint(file):
    checkpoint = torch.load(file)
    if checkpoint['Architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    if checkpoint['Architecture'] == 'resnet101' :
        model = models.resnet101(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

#Process Image
def process_image(image):
    pil_image = Image.open(image)
    
    image_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    new_im = image_transforms(pil_image)
    np_image = np.array(new_im)
        
    return new_im 

#Predict image
def predict(model, topk, image_path):
    model.cpu()
    model.eval()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    output = process_image(image_path)
    output.unsqueeze_(0)
    log_ps = model.forward(output)
    ps = torch.exp(log_ps)
    probs, labels = ps.topk(topk, dim=1)
    
    inverted_class_to_idx = dict(map(reversed, model.class_to_idx.items()))
    
    probs = np.array(probs.detach())[0] 
    labels = np.array(labels.detach())[0]
    
    labels = [inverted_class_to_idx[lab] for lab in labels]
    flowers = [cat_to_name[lab] for lab in labels]
    
    return probs, labels, flowers
   