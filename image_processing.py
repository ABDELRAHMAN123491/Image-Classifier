'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window.

'''


from PIL import Image
from torchvision import transforms
import numpy as np


def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(img_path)

    # Defining the preprocess transforms without Normalization
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    img_tensor = preprocess(img_pil)

    # Normalization
    np_image = np.array(img_tensor) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    return np_image
