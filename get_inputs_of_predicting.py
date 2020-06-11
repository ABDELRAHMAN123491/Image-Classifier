'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window.

'''

import argparse


def predicting_inputs():
    """
    Command Line Arguments:
    1. Image path as --img_path with default value as './flowers/test/1/image_06764.jpg'
    2. The checkpoint as --checkpoint with default value as './checkpoint.pth'
    3. Processing usnit as --gpu with defual value as 'gpu'
    4. Top classes in terms of the probabilities as --top_classes with default value as 5
    5. Cat_to_name as --cat_to_name with default value as 'cat_to_name.json'

    This function returns these arguments as an ArgumentParser object.

    Parameters:
    None

    Returns:
    parse_args()

    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', dest='img_path', action='store',
                        default="./flowers/test/10/image_07090.jpg")
    parser.add_argument('--gpu', dest='gpu', action='store', default="gpu")
    parser.add_argument('--checkpoint', dest='checkpoint', action="store",
                        default="./checkpoint.pth")
    parser.add_argument('--top_classes', type=int,
                        dest="top_classes", action="store", default=5)
    parser.add_argument('--category_names', action="store",
                        dest="category_names",
                        default="cat_to_name.json")

    return parser.parse_args()
