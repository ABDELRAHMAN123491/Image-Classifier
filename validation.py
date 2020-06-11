'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:
    Validation on the test dataset

'''
import torch


def test_validation(model, testloader, processor='gpu'):
    """
    This function validates the model on the test dataset

    Parameters:
     model, testloader, processor='gpu'

    Returns:
     None

    """
    if processor == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    accuracy = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)

            # Acuuracy Calculations
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print("Test Accuracy: {:.3f}".format(accuracy / len(testloader)))
