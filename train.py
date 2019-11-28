# imports
from functions import *

#Loading in the data
trainloader, testloader, validloader, train_data = load_data(args.data_dir)
print(trainloader)
#Creating model structure
cat_to_name, model, criterion, optimizer = model_stuff()
#print(model)

#Training the model
training(trainloader,optimizer,model, criterion, validloader)

#Saving the model
saving(args.arch, model, args.epochs,train_data)