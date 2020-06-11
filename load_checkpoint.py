'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:
    loading the checkpoint

'''

import torch
from classifier import classifier

in_features = {'vgg16': 25088,
               'alexnet': 9216,
               "vgg13": 25088}


def load_checkpoint(path='./checkpoint.pth'):
    """
    This function loads the checkpoint

    Parameters:
     path='./checkpoint.pth'

    Returns:
     model

    """
    checkpoint = torch.load(path)

    # Retrieving information from the checkpoint
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    learning_rate = checkpoint['learning_rate']

    model, __, __ = classifier(arch, dropout,
                               hidden_units, learning_rate)

    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']

    model.load_state_dict(checkpoint['model_state_dict'])
    return model
