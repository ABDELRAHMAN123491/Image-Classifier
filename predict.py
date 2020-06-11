'''
Developer's name: ABDELRAHMAN S. ABDELRAHMAN

Date: 10th of June 2020

Purpose:

Training the model based on the options that user determine

'''
import json
from get_inputs_of_predicting import predicting_inputs
from load_checkpoint import load_checkpoint
from Predicting import predict


def main():
    """
    This function comprises all other required function to do
    prediction

    Parameters:
     None

    Returns:
     None
    """

    in_arg = predicting_inputs()
    img_path = in_arg.img_path
    device = in_arg.gpu
    num_of_top_probabilities = in_arg.top_classes
    load_check_path = in_arg.checkpoint
    cat_to_name = in_arg.category_names

    # Loading checkpoint
    model = load_checkpoint(load_check_path)

    # Image_processing and Predicting
    top_p, top_classes = predict(img_path, model,
                                 num_of_top_probabilities, device)

    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    # Obtaining the names from the classes
    Flower_names = [cat_to_name[i] for i in top_classes]

    print(top_p)
    print(top_classes)

    print("The image you entered belongs to {}, with a proability of {:.3f}".format(Flower_names[0], top_p[0]))


if __name__ == '__main__':
    main()
