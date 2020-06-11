'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:

Training the model based on the options that user determine

'''


from get_inputs_of_training import training_inputs
from transformed_loaded_data import transformed_loaded_data
from classifier import classifier
from training import training_network
from validation import test_validation
from Save_checkpoint import save_checkpoint


def main():
    """
    This function comprises all other required function to do
    training

    Parameters:
     None

    Returns:
     None

    """
    in_arg = training_inputs()

    data_dir = in_arg.dir_data
    device = in_arg.gpu
    epochs = in_arg.epochs
    archeticture = in_arg.arch
    dropout = in_arg.dropout
    hidden_units = in_arg.hidden_units
    lr = in_arg.learning_rate
    saving = in_arg.save_dir

    # Loading data
    trainloader, validloader, testloader, train_data = transformed_loaded_data(data_dir)

    # Building the Classifier
    model, criterion, optimizer = classifier(archeticture, dropout,
                                             hidden_units, lr, device)

    # Training
    training_network(model, optimizer, criterion,
                     trainloader, validloader, epochs, device)

    # Validation
    test_validation(model, testloader, device)

    # Saving the Checkpoint
    save_checkpoint(model, train_data, optimizer, archeticture, epochs,
                    dropout, lr,
                    hidden_units, saving)

    print("__________Well Done_____________")
    print("__________Now you can use the model for prediction_____________")


if __name__ == '__main__':
    main()
