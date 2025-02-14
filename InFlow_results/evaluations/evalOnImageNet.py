import torch
import torch.nn as nn
from torchvision import transforms
import csv
import argparse
import time
import numpy as np
from PIL import Image
import os
import warnings

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from util.test_methods import MASTestFunctions as MAS
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

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

resize = transforms.Resize((224, 224), antialias = True)

def run_and_save_tests(img_hw, image_count, function, transform, model, explainer, LRP_explainer, model_name, device, imagenet, batch_size):
    # num imgs used for testing
    img_label = str(image_count) + "_images_"

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    num_classes = 1000
    images_per_class = int(np.ceil(image_count / num_classes))
    classes_used = [0] * num_classes

    fields = ["attr", "Rise Ins", "Rise Del", "Rise Ins - Del", "MAS Ins", "MAS Del", "MAS Ins - Del", "time mean", "time SD"]
    scores = [function, 0, 0, 0, 0, 0, 0, 0, 0]
    total_time = []

    images = sorted(os.listdir(imagenet))
    images_used = 0

    klen = 11
    ksig = 11
    kern = MAS.gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)
    MAS_insertion = MAS.MASMetric(model, img_hw * img_hw, 'ins', img_hw, substrate_fn = blur)
    MAS_deletion = MAS.MASMetric(model, img_hw * img_hw, 'del', img_hw, substrate_fn = torch.zeros_like)

    # look at test images in order from 1
    for image in images:    
        if images_used == image_count:
            print("method finished")
            break

        begin = time.time()

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
        original_pred = model_utils.getPrediction(img_tensor, model, device, target_class)[0] * 100
        if original_pred < 60:
            continue
        blur_pred = model_utils.getPrediction(blur(img_tensor.cpu()).to(device), model, device, target_class)[0] * 100
        if blur_pred >= original_pred:
            continue

        # Track which classes have been used
        if classes_used[target_class] == images_per_class:
            continue
        else:
            classes_used[target_class] += 1       

        print(model_name + " Function " + function + ", image: " + image)

        if (function == "Raw_Attn"):
            start = time.time()
            attn = explainer.generate_raw_attn(img_tensor, device = device)
            attr = resize(attn.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'LRP'):
            start = time.time()
            attr = LRP_explainer.generate_LRP(img_tensor, target_class, method="transformer_attribution", start_layer = 0, device = device)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'LRP_IG'):
            start = time.time()
            _, IG, _, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
            IG = resize(IG.cpu().detach())
            attr = LRP_explainer.generate_LRP(img_tensor, target_class, method="transformer_attribution", start_layer = 0, device = device, withgrad = False)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr * IG).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'Naive_Rollout'):
            start = time.time()
            attr, _, _ = explainer.generate_naive_rollout(img_tensor, start_layer = 0)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'Naive_Rollout_IG'):
            start = time.time()
            _, IG, _, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
            IG = resize(IG.cpu().detach())
            attr, _, _ = explainer.generate_naive_rollout(img_tensor, start_layer = 0)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr * IG).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'Rollout'):
            start = time.time()
            attr, _, _ = explainer.generate_rollout(img_tensor, start_layer = 0)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'Rollout_IG'):
            start = time.time()
            _, IG, _, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
            IG = resize(IG.cpu().detach())
            attr, _, _ = explainer.generate_rollout(img_tensor, start_layer = 0)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr * IG).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'Transition_attn_MAP'):
            start = time.time()
            map, _, _, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
            map = resize(map.cpu().detach())
            saliency_map = (map).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'Transition_attn'):
            start = time.time()
            _, _, attr, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'Bidirectional_MAP'):
            start = time.time()
            _, attr = explainer.bidirectional(img_tensor, target_class, device = device)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'Bidirectional'):
            start = time.time()
            attr, _ = explainer.bidirectional(img_tensor, target_class, device = device)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0)
            measured_time = time.time() - start

        elif(function == 'InFlow'):
            start = time.time()
            InFlow, _ = explainer.generate_InFlow(img_tensor, target_class, device = device, option = "b")
            InFlow = resize(InFlow.cpu().detach())
            saliency_map = (InFlow).permute(1, 2, 0)
            measured_time = time.time() - start
        elif (function == 'InFlow_1'):
            start = time.time()
            InFlow, _ = explainer.generate_InFlow(img_tensor, target_class, device = device, withgrad = False, option = "b", ablate = 1)
            InFlow = resize(InFlow.cpu().detach())
            saliency_map = (InFlow).permute(1, 2, 0)
            measured_time = time.time() - start
        elif(function == 'InFlow_2'):
            start = time.time()
            InFlow, _ = explainer.generate_InFlow(img_tensor, target_class, device = device, withgrad = False, option = "b")
            InFlow = resize(InFlow.cpu().detach())
            saliency_map = (InFlow).permute(1, 2, 0)
            measured_time = time.time() - start

        elif (function == 'InFlow_IG'):
            start = time.time()
            _, IG, _, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
            IG = resize(IG.cpu().detach())
            InFlow, _ = explainer.generate_InFlow(img_tensor, target_class, device = device, option = "b", withgrad = False)
            InFlow = resize(InFlow.cpu().detach())
            saliency_map = (InFlow * IG).permute(1, 2, 0)
            measured_time = time.time() - start
        else:
            print("You have not picked a valid attribution method.")

        # make sure attribution is valid
        if np.sum(saliency_map.numpy().reshape(1, 1, img_hw ** 2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue

        total_time.append(measured_time)

        # Get attribution scores
        saliency_map_test = np.abs(np.sum(saliency_map.numpy(), axis = 2, keepdims=True))

        ins_del_img = img_tensor.cpu()
        _, corrected_score_i, _, _, raw_score_i = MAS_insertion.single_run(ins_del_img, saliency_map_test, device, max_batch_size = batch_size)
        _, corrected_score_d, _, _, raw_score_d = MAS_deletion.single_run(ins_del_img, saliency_map_test, device, max_batch_size = batch_size)

        scores[1] += MAS.auc(raw_score_i)
        scores[2] += MAS.auc(raw_score_d)
        scores[3] += MAS.auc(raw_score_i) - MAS.auc(raw_score_d)
        scores[4] += MAS.auc(corrected_score_i)
        scores[5] += MAS.auc(corrected_score_d)
        scores[6] += MAS.auc(corrected_score_i) - MAS.auc(corrected_score_d)

        # when all tests have passed, the number of images used can go up by 1
        images_used += 1

        print("Total used: " + str(images_used) + " / " + str(image_count))

        print(time.time() - begin)

    total_time = np.array(total_time)
    avg = round(np.mean(total_time), 5)
    std = round(np.std(total_time), 5)
    scores[9] = avg
    scores[10] = std

    for i in range(1, 9):
        scores[i] /= images_used
        scores[i] = round(scores[i], 3)

    # make the test folder if it doesn't exist
    folder = "../test_results/" + model_name + "/" + "imagenet/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    img_label = function + "_" + str(image_count) + "_images"
    with open(folder + img_label + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerow(scores)

    return

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.gpu) if torch.cuda.is_available() else 'cpu'

    model_name = FLAGS.model

    batch_size = 15

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

    run_and_save_tests(img_hw, FLAGS.image_count, FLAGS.function, transform, model, explainer, LRP_explainer, model_name, device, FLAGS.imagenet, batch_size)

if __name__ == "__main__":
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Attribution Test Script.')
    parser.add_argument('--function',
                        type = str, default = "IG",
                        help = 'Name of the attribution method to use: .')
    parser.add_argument('--model',
                        type = str, default = "VIT_base_16",
                        help = 'Name of the model to use: VIT_tiny_16, VIT_base_32, VIT_base_16.')
    parser.add_argument('--image_count',
                        type = int, default = 5000,
                        help='How many images to test with.')
    parser.add_argument('--gpu',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--imagenet',
                type = str, default = "imagenet",
                help = 'The path to your 2012 imagenet validation set. Images in this folder should have the name structure: "ILSVRC2012_val_00000001.JPEG".')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)