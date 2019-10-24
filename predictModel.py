from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    start_width = image.size[0]
    start_height = image.size[1]
    
    if start_width > start_height:
        image.thumbnail((100000, 256))

    else:
        image.thumbnail((256, 100000))
    
    width, height = image.size   

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    image = image.crop((left, top, right, bottom))
    np_image = np.array(image)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image-mean)/std

    np_image = np_image.transpose((2,0,1))

    return torch.tensor(np_image)

def predict(image_path, checkpoint, topk, cat_names_path, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    model = models.densenet121(pretrained=True)
    
    
    for param in model.parameters():
        param.requires_grad = False
    
    with open(cat_names_path, 'r') as f:
        cat_to_name = json.load(f)
    
    model.class_to_idx = cat_to_name
    
#     checkpoint_data = torch.load(checkpoint)
    
#     model.classifier = checkpoint_data['classifier']
#     model.classifier = nn.Sequential(nn.Linear(1024, checkpoint_data['hidden_layer1']),
#                                  nn.ReLU(),
#                                  nn.Dropout(0.2),
#                                  nn.Linear(checkpoint_data['hidden_layer1'], 102),
#                                  nn.LogSoftmax(dim=1))
    
 
#     model.load_state_dict(checkpoint_data['state_dict'])
    
    checker = torch.load('checkpoint.pth')
    model.classifier = nn.Sequential(nn.Linear(1024, checker['hidden_layer1']),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(checker['hidden_layer1'], 102 ),
                                 nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checker['state_dict'])
    
    
    
    
    image = Image.open(image_path)
   
    
    image = process_image(image)
    image.unsqueeze_(0)
    
    
    with torch.no_grad():
        model.eval()
        image = image.to(device)
#         model.to(device)
#         print(type(image))
        ouput = model.forward(image.type(torch.FloatTensor))
        ps = torch.exp(ouput)
        top_p, top_class = ps.topk(topk, dim=1)
    print(top_p, [model.class_to_idx[str(cat)] for cat in top_class.numpy()[0]])
    return top_p, [model.class_to_idx[str(cat)] for cat in top_class.numpy()[0]]