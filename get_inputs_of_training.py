'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:
    Retrieves and parses the 8 command line arguments provided by the user when
    they run the program from a terminal window.

'''

import argparse


def training_inputs():
    """
    Command Line Arguments:
      1. Flower Folder as --dir with default value './flowers'
      2. The processing unit as --gpu with default value as 'gpu'
      3. The number of epochs as --epochs with default value as '15'
      4. The CNN archeticture as --arch with default value as 'vgg16'
      5. Saving directory as --save_dir with default value as './checkpoint.pth'
      6. Learning rate as --learning_rate with default value as '0.001'
      7. Dropout as --dropout with default value as '0.2'
      8. Hidden units as --hidden_unit with default value as '5000'

    This function returns these arguments as an ArgumentParser object.

    Parameters:
     None

    Returns:
     parse_args()

    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('dir_data', type=str,
                        action='store', default="./flowers")
    parser.add_argument('--gpu', dest='gpu', action='store', default="gpu")
    parser.add_argument('--epochs', dest="epochs",
                        type=int, default=20)
    parser.add_argument('--arch', dest="arch", action="store",
                        default="vgg16", type=str)
    parser.add_argument('--save_dir', dest="save_dir", action="store",
                        default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", type=float, default=0.001)
    parser.add_argument('--dropout', dest="dropout",
                        action="store", type=float, default=0.2)
    parser.add_argument('--hidden_units', type=int,
                        dest="hidden_units", action="store", default=5000)

    return parser.parse_args()
