#Imports
import numpy as np
from functions_predict import *

#Load checkpoint
model = load_checkpoint(args.checkpoint)

#Process image
output = process_image(args.image_path)
output = output.numpy()

#Probs of classes and flowers
probs, labels, flowers = predict(model, args.top_k, args.image_path)

#Final output
print('The model predicts this flower to be a' ,flowers[0] , 'with probability of' , probs[0])

if args.top_k > 1:
    print('The top', args.top_k, 'classes are', flowers, 'and their probabilities are', probs, 'respectively.')