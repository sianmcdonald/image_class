#Imports
import torch
from torchvision import transforms, models
import json
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--image_path', type=str, default = './flowers/test/18/image_04277.jpg', help='Path for image to predict')
parser.add_argument('--top_k', type=int, default = 1, help='Top k predictions for image')
parser.add_argument('--category_names', type=str, default = "cat_to_name.json", help='Maps categorys to names')
parser.add_argument('--gpu', type=str, default = "cuda", help='Use GPU or not - input cude or cpu')
parser.add_argument('--checkpoint', type=str, default ='checkpoint.pth', help = 'Checkpoint file directory')
args = parser.parse_args()

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
   