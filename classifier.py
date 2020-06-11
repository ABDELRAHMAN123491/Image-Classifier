'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:
    Building the classifier parameters

'''

import torch
from torch import nn, optim
from torchvision import models


in_features = {'vgg16': 25088,
               'alexnet': 9216,
               "vgg13": 25088}


def classifier(archeticture='vgg16', dropout=0.2, hidden_units=5000, learning_rate=0.001, processor='gpu'):
    """
    This function builds the classifier according to the user's specifications

    Parameters:
     archeticture='vgg16', dropout=0.2, hidden_units=5000, lr=0.001, device='gpu'

    Returns:
     model, criterion, optimizer

    """
    if processor == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    # archeticture Type
    if archeticture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif archeticture == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif archeticture == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print("The archeticture {} is not supported, type vgg16, vgg13, or alexnet".format(archeticture))
        print("We will use the default one, which is vgg16")
        model = models.vgg16(pretrained=True)

    # Customize the classifier of the pretrained Network
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(in_features[archeticture], hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Training the classifier parameters only
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    return model, criterion, optimizer
