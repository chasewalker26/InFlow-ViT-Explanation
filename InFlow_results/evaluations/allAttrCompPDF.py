from fpdf import FPDF
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import argparse
import warnings

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util.visualization import attr_to_subplot
from util import model_utils
from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines, LRP

# models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_tiny_patch16_224 as vit_new_tiny_16
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_tiny_patch16_224 as vit_LRP_tiny_16

    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224 as vit_new_base_32
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch32_224 as vit_LRP_base_32

    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch16_224 as vit_new_base_16
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch16_224 as vit_LRP_base_16

model = None

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

resize = transforms.Resize((224, 224), antialias = True)

def run_and_save_tests(img_hw, transform, model, explainer, LRP_explainer, model_name, device, imagenet):
    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    functions = ["Attn", "GC", "IG", "N-Roll", "Rollout",  "Bi-Attn", "T-Attn", "T-Attr", "InFlow"]

    images_used = 0
    images_seen = 0
    images_per_page = 16
    number_of_pages = 4
    total_images = images_per_page * number_of_pages
    images_per_class = int(np.ceil(total_images / 1000))
    classes_used = [0] * 1000

    # make the temp image folders if they don't exist
    if not os.path.exists("../temp_folder_attr/images"):
        os.makedirs("../temp_folder_attr/images")

    pdf = FPDF(format = "letter", unit="in")

    # classes excluded due to undesirable input images
    excluded_classes = [434, 435, 436, 638, 639, 842]

    images = sorted(os.listdir(imagenet))

    while (images_used != total_images):
        fig, axs = plt.subplots(images_per_page, len(functions) + 1, figsize = ((len(functions) + 1) * 5, images_per_page * 5.6))
        plt.rcParams.update({'font.size': 45})

        i = 0
        while (i != images_per_page):
            image = images[images_seen]
            images_seen += 1

            # check if the current image is an invalid image for testing, 0 indexed
            image_num = int((image.split("_")[2]).split(".")[0]) - 1
            # check if the current image is an invalid image for testing
            if correctly_classified[image_num] == 0:
                continue

            image_path = imagenet + "/" + image
            PIL_img = Image.open(image_path)
            img = transform(PIL_img)
            img_tensor = normalize(img).unsqueeze(0).to(device)

            # only rgb images can be classified
            if img.shape != (3, img_hw, img_hw):
                continue

            target_class = model_utils.getClass(img_tensor, model, device)
            if target_class in excluded_classes:
                continue

            with open('../../util/class_maps/ImageNet/imagenet_classes.txt') as f:
                classes = [line.strip() for line in f.readlines()]
            class_name = classes[target_class]

            # Track which classes have been used
            if classes_used[target_class] == images_per_class:
                continue
            else:
                classes_used[target_class] += 1       

            axs[i, 0].set_ylabel(class_name.replace(" ", "\n"), fontsize = 45)
            attr_to_subplot(img, "Input", axs[i, 0], original_image = True)           
            
            images_used += 1
            j = 1
            print(model_name + ", image: " + image + " " + str(images_used) + "/" + str(total_images))

            for function in functions:
                if (function == "Attn"):
                    attn = explainer.generate_raw_attn(img_tensor, device = device)
                    attr = resize(attn.cpu().detach())
                    saliency_map = (attr).permute(1, 2, 0)
                elif (function == 'GC'):    
                    attr = explainer.generate_cam_attn(img_tensor, target_class, device = device)
                    attr = resize(attr.cpu().detach())
                    saliency_map = attr.permute(1, 2, 0)
                elif (function == 'IG'):    
                    _, IG, _, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
                    IG = resize(IG.cpu().detach())
                    saliency_map = IG.permute(1, 2, 0)    
                elif (function == 'N-Roll'):
                    attr, _, _ = explainer.generate_naive_rollout(img_tensor, start_layer = 0)
                    attr = resize(attr.cpu().detach())
                    saliency_map = (attr).permute(1, 2, 0)
                elif (function == 'Rollout'):
                    attr, _, _ = explainer.generate_rollout(img_tensor, start_layer = 0)
                    attr = resize(attr.cpu().detach())
                    saliency_map = (attr).permute(1, 2, 0)
                elif (function == 'Bi-Attn'):
                    attr, _ = explainer.bidirectional(img_tensor, target_class, device = device)
                    attr = resize(attr.cpu().detach())
                    saliency_map = (attr).permute(1, 2, 0)
                elif (function == 'T-Attn'):
                    _, _, attr, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
                    attr = resize(attr.cpu().detach())
                    saliency_map = (attr).permute(1, 2, 0)
                elif (function == 'T-Attr'):
                    attr = LRP_explainer.generate_LRP(img_tensor, target_class, method="transformer_attribution", start_layer = 0, device = device)
                    attr = resize(attr.cpu().detach())
                    saliency_map = (attr).permute(1, 2, 0)
                elif(function == 'InFlow'):
                    InFlow, _ = explainer.generate_InFlow(img_tensor, target_class, device = device, option = 'b')
                    InFlow = resize(InFlow.cpu().detach())
                    saliency_map = (InFlow).permute(1, 2, 0)
                
                if torch.isnan(saliency_map).any():
                    classes_used[target_class] -= 1
                    images_used -= 1
                    i -= 1
                    break
                    
                # put image and attributions into the plot column i
                attr_to_subplot(saliency_map.numpy(), function, axs[i, j], cmap = 'jet', norm = 'absolute', blended_image = img)

                j += 1
            i += 1

        # save the figure for the current attribution function
        fig.tight_layout()
        plt.figure(fig)
        plt.subplots_adjust(hspace = 0.12, wspace = 0.05)
        plt.savefig("../temp_folder_attr/images/" + str(images_used + 1 - images_per_page) + "_" + str(images_used) + ".png", dpi = 25, bbox_inches='tight', transparent = "False", pad_inches = .05)
        fig.clear()
        plt.close(fig)

    print("Building PDF")
    for file in sorted(os.listdir("../temp_folder_attr/images")):
        # (LM x TM x W x H)
        pdf.add_page()
        pdf.image("../temp_folder_attr/images/" + file, 1.5, 1, 5.5, 8.8)

    pdf.output("../test_results/" + model_name + "_comps" + ".pdf", "F")
    pdf.close()

    # clear the folder that held the images
    print("Clearing files")
    dir = '../temp_folder_attr/images'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    os.rmdir(dir)

    os.rmdir('../temp_folder_attr')

    print("Done")

    return

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    model_name = FLAGS.model

    if model_name == "VIT_tiny_16":
        model = vit_new_tiny_16(pretrained=True).to(device).eval()
        model_lrp = vit_LRP_tiny_16(pretrained=True).to(device).eval()
    elif model_name == "VIT_base_32":
        model = vit_new_base_32(pretrained=True).to(device).eval()
        model_lrp = vit_LRP_base_32(pretrained=True).to(device).eval()
    elif model_name == "VIT_base_16":
        model = vit_new_base_16(pretrained=True).to(device).eval()
        model_lrp = vit_LRP_base_16(pretrained=True).to(device).eval()

    explainer = Baselines(model)
    LRP_explainer = LRP(model_lrp)


    img_hw = 224
    transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    run_and_save_tests(img_hw, transform, model, explainer, LRP_explainer, model_name, device, FLAGS.imagenet)

if __name__ == "__main__":
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Make a pdf comparing attribution methods')
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--model',
                        type = str, default = "VIT_base_16",
                        help = 'Name of the model to use: VIT_tiny_16, VIT_base_32, VIT_base_16.')
    parser.add_argument('--imagenet',
                type = str, default = "../../ImageNet",
                help = 'The path to your 2012 imagenet validation set. Images in this folder should have the name structure: "ILSVRC2012_val_00000001.JPEG".')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)