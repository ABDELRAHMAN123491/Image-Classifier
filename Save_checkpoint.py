'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:
    saving the checkpoint

'''
import torch


def save_checkpoint(model, train_data, optimizer, arch='vgg16', epochs=15,
                    dropout=0.2, learning_rate=0.001,
                    hidden_units=5000, directory='./checkpoint.pth'):
    """
    This function saves the checkpoint for prediction

    Parameters:
     model, testloader, processor='gpu'

    Returns:
     None

    """
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': arch,
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'hidden_units': hidden_units,
                  'dropout': dropout, 'learning_rate': learning_rate,
                  'epochs': epochs}

    torch.save(checkpoint, directory)
