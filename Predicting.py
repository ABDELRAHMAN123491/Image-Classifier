'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:
    Predicting an image

'''


import torch
from image_processing import process_image


def predict(image_path, model, topk=5, processor='gpu'):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    if processor == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    model.to(device)

    # image processing
    image = process_image(image_path)

    # Convert the np.array to a Pytorch tensor
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)

    # Add dimension for batch
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_idx = ps.topk(topk, dim=1)

    model.train()

    # Convert to list

    top_p = top_p.detach().type(torch.FloatTensor).numpy().tolist()[0]

    top_idx = top_idx.detach().type(torch.FloatTensor).numpy().tolist()[0]

    # invert the dictionary 'class_to_idx' to obtain the classes of the top_probabilities via top indices

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}

    top_classes = [idx_to_class[idx] for idx in top_idx]

    return top_p, top_classes
