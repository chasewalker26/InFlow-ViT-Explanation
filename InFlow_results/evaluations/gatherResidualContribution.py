import torch
from torchvision import transforms
import csv
import argparse
import time
import numpy as np
from PIL import Image
import os

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils

from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines
from util.attribution_methods.VIT_LRP.ViT_new import vit_base_patch16_224 as vit_new

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

resize = transforms.Resize((224, 224), antialias = True)

# runs an attribution method w 3 baselines over imageCount images and calculates the mean PIC
def run_and_save_tests(img_hw, image_count, transform, model, explainer, model_name, device, imagenet):
    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    num_classes = 1000
    images_per_class = int(np.ceil(image_count / num_classes))
    classes_used = [0] * num_classes

    images = sorted(os.listdir(imagenet))
    images_used = 0
    
    resid_1_ratio = 0
    resid_2_ratio = 0

    # look at test images in order from 1
    for image in images:    
        if images_used == image_count:
            print("method finished")
            break

        # check if the current image is an invalid image for testing, 0 indexed
        image_num = int((image.split("_")[2]).split(".")[0]) - 1
        # check if the current image is an invalid image for testing
        if correctly_classified[image_num] == 0:
            continue

        img = Image.open(imagenet + "/" + image)
        trans_img = transform(img)

        # put the image in form needed for prediction for the ins/del method
        img_tensor = transform_normalize(trans_img)
        img_tensor = torch.unsqueeze(img_tensor, 0).to(device)

        # only rgb images can be classified
        if trans_img.shape != (3, img_hw, img_hw):
            continue

        target_class = model_utils.getClass(img_tensor, model, device)

        # Track which classes have been used
        if classes_used[target_class] == images_per_class:
            continue
        else:
            classes_used[target_class] += 1       

        _, biases = explainer.generate_InFlow(img_tensor, target_class, start_layer = 0, end_layer = 12, version = 4, device = device)

        r1_biases = biases[0]
        r2_biases = biases[1]

        resid_1_ratio += (r1_biases[:, 0].reshape(12, 197) / r1_biases[:, 1].reshape(12, 197)).detach().cpu().numpy()
        resid_2_ratio += (r2_biases[:, 0].reshape(12, 197) / r2_biases[:, 1].reshape(12, 197)).detach().cpu().numpy()

        # when all tests have passed, the number of images used can go up by 1
        images_used += 1

        print("Total used: " + str(images_used) + " / " + str(image_count))

    # make the test folder if it doesn't exist
    folder = "../test_results/" + model_name + "/" + "imagenet/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    img_label = "resid_ratio_" + str(image_count) + "_images"
    with open(folder + img_label + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerows(resid_1_ratio / image_count)
        write.writerows(resid_2_ratio / image_count)

    return

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.gpu) if torch.cuda.is_available() else 'cpu'

    model_name = "VIT_base_16"

    model = vit_new(pretrained=True).to(device)
    model = model.eval()

    explainer = Baselines(model)

    img_hw = 224
    transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    run_and_save_tests(img_hw, FLAGS.image_count, transform, model, explainer, model_name, device, FLAGS.imagenet)

if __name__ == "__main__":
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_count',
                        type = int, default = 10000,
                        help='How many images to test with.')
    parser.add_argument('--gpu',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--imagenet',
                type = str, default = "imagenet",
                help = 'The path to your 2012 imagenet validation set. Images in this folder should have the name structure: "ILSVRC2012_val_00000001.JPEG".')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)