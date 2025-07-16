# https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_explanation_generator.py

import argparse
import torch
import numpy as np
from numpy import *
import torch.nn as nn

# compute rollout between attention layers, no residual connection
def compute_rollout_naive(all_layer_matrices, start_layer = 0):
    all_layer_matrices = torch.stack(all_layer_matrices)
    num_blocks = all_layer_matrices.shape[0]

    # perform rollout with no residual connection
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, num_blocks):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)

    return joint_attention, all_layer_matrices

# compute rollout between attention layers with residual connection modeling
# https://github.com/samiraabnar/attention_flow
def compute_rollout_attention(all_layer_matrices, start_layer = 0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)

    # add residual connection modeling to all matrices
    all_layer_matrices = torch.stack(all_layer_matrices)
    num_blocks = all_layer_matrices.shape[0]

    matrices_aug = all_layer_matrices + eye.unsqueeze(0)

    # normalize all the matrices, making residual connection addition equal to 0.5*A + 0.5*I
    matrices_aug = matrices_aug / matrices_aug.sum(dim=-1, keepdim=True)

    # perform rollout
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, num_blocks):
        joint_attention = matrices_aug[i].bmm(joint_attention)

    return joint_attention, matrices_aug

# compute rollout between attention layers with full transformer block residual modeling
def compute_InFlow(all_layer_attentions, all_layer_biases_resid_1, all_layer_biases_resid_2, ablate = 0):
    batch_size = all_layer_attentions[0].shape[0]
    num_tokens = all_layer_attentions[0].shape[1]
    eye = torch.eye(num_tokens).expand(1, batch_size, num_tokens, num_tokens).to(all_layer_attentions[0].device)

    # add residual connection modeling to all matrices
    all_layer_attentions = torch.stack(all_layer_attentions)
    all_layer_biases_resid_1 = torch.stack(all_layer_biases_resid_1).to(all_layer_attentions[0].device)
    all_layer_biases_resid_2 = torch.stack(all_layer_biases_resid_2).to(all_layer_attentions[0].device)

    num_blocks = all_layer_attentions.shape[0]

    # residual connection formula
    # residual = data * main_path_norm + I * skip_connection_norm

    # Standard InFlow
    if ablate == 0:
        # model the first residual connection
        # A * Attn_bias + I * Input_bias
        matrices_aug_resid_1 = all_layer_attentions * all_layer_biases_resid_1[:, 1].reshape(num_blocks, 1, 1, num_tokens) + eye * torch.diag_embed(all_layer_biases_resid_1[:, 0].squeeze()).reshape(all_layer_attentions.shape)
        # model the second residual connection
        # meaure how much the mlp layers scale the first residual connection result
        mlp_resid_1_ratio = all_layer_biases_resid_2[:, 1].reshape(num_blocks, 1, 1, num_tokens) / all_layer_biases_resid_2[:, 0].reshape(num_blocks, 1, 1, num_tokens)
        mlp_resid_1_ratio = torch.nn.functional.normalize(mlp_resid_1_ratio, p = 1, dim = -1)
        # MLP_scaling_ratio * MLP_bias + I * resid_1_bias
        matrices_aug_resid_2 = torch.diag_embed(mlp_resid_1_ratio.squeeze()).reshape(all_layer_attentions.shape) * torch.diag_embed(all_layer_biases_resid_2[:, 1].squeeze()).reshape(all_layer_attentions.shape) + eye * torch.diag_embed(all_layer_biases_resid_2[:, 0].squeeze()).reshape(all_layer_attentions.shape)
        # multiply the residual connections
        matrices_aug = matrices_aug_resid_1 @ matrices_aug_resid_2
    # InFlow, just first residual, aA + bI
    elif ablate == 1:
        matrices_aug_resid_1 = all_layer_attentions * all_layer_biases_resid_1[:, 1].reshape(num_blocks, 1, 1, num_tokens) + eye * torch.diag_embed(all_layer_biases_resid_1[:, 0].squeeze()).reshape(all_layer_attentions.shape)
        matrices_aug = matrices_aug_resid_1
    
    # normalize the scale of the matrices
    matrices_aug = matrices_aug / matrices_aug.sum(dim = -1, keepdim = True)

    joint_attention = matrices_aug[0]

    for i in range(1, num_blocks):
        joint_attention = matrices_aug[i].bmm(joint_attention)

    return joint_attention, matrices_aug

def generate_attn_grad(model, input, target_class, target_block = -1):
    output = model(input.cuda(), register_hook=True)
    score = output[0][target_class].sum()
    score.backward()

    grad = model.blocks[target_block].attn.get_attn_gradients().mean(1)
    grad = grad[0, 0, 1:]
    patches_per_side = int(np.sqrt(grad.shape[-1]))
    return grad.reshape(-1, patches_per_side, patches_per_side).detach()

def getPrediction(input, model, target_class):
    output = model(input)
    scores = output[:, target_class].detach()
    return scores.squeeze()

# https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_explanation_generator.py
class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, target_class, method="transformer_attribution", is_ablation=False, start_layer=0, withgrad = True, device = "cuda:0"):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if target_class == None:
            target_class = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, target_class] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        num_blocks = len(self.model.blocks)

        attr = self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, end_layer=num_blocks, withgrad=withgrad, **kwargs)
        
        patches_per_side = int(np.sqrt(attr.shape[-1]))

        return attr.reshape(-1, patches_per_side, patches_per_side)

class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_raw_attn(self, input, device = "cuda:0"):
        self.model(input.to(device))
        attn = self.model.blocks[-1].attn.get_attention_map().mean(1)
        attn = attn[0, 0, 1:]
        patches_per_side = int(np.sqrt(attn.shape[-1]))
        return attn.reshape(-1, patches_per_side, patches_per_side)

    # https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_explanation_generator.py
    def generate_cam_attn(self, input, target_class, device = "cuda:0"):
        input.requires_grad = True
        output = self.model(input.to(device), register_hook=True)
        score = output[0][target_class].sum()
        score.backward()
        input.requires_grad = False

        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        patches_per_side = int(np.sqrt(grad.shape[-1] - 1))
        cam = self.model.blocks[-1].attn.get_attention_map()[0, :, 0, 1:].reshape(-1, patches_per_side, patches_per_side)

        grad = grad[0, :, 0, 1:].reshape(-1, patches_per_side, patches_per_side)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.unsqueeze(0)
        #################### attn

    def generate_naive_rollout(self, input, start_layer = 0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)

        rollout, test = compute_rollout_naive(all_layer_attentions, start_layer = start_layer)
        rollout = rollout[:, 0, 1:]
        patches_per_side = int(np.sqrt(rollout.shape[-1]))

        return rollout.reshape(-1, patches_per_side, patches_per_side), test, torch.stack(all_layer_attentions)

    def generate_rollout(self, input, start_layer = 0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []

        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)

        rollout, test = compute_rollout_attention(all_layer_attentions, start_layer = start_layer)

        rollout = rollout[:, 0, 1:]
        patches_per_side = int(np.sqrt(rollout.shape[-1]))

        return rollout.reshape(-1, patches_per_side, patches_per_side), test, torch.stack(all_layer_attentions)

    def generate_InFlow(self, input, target_class, withgrad = True, device = "cuda:0", ablate = 0, stop_layer = 12, target_token = 0):
        '''
        The user may set two additional paramaters to control the explanation:
        stop layer: the layer up to which InFlow will be performed 
        target_token: The InFlow operation produces a matrix of [tokens, tokens].
                      The standard operation is to take the first row (the [CLS] token).
                      The user may select another token if they see fit, but this is
                      **NOT RECOMMENED** as it is unlikely to provide a strong 
                      explanation, since only the [CLS] token captures global image information.
        '''
        blocks = self.model.blocks
        _, num_heads, _, _ = self.model.blocks[-1].attn.get_attention_map().shape

        all_layer_attentions = []
        all_layer_biases_resid_1 = []
        all_layer_biases_resid_2 = []

        input.requires_grad = True
        output = self.model(input.to(device), register_hook=True)
        score = output[0][target_class].sum()
        score.backward(retain_graph=True)
        input.requires_grad = False

        prob = self.model.get_block_classification_probs()

        for i, blk in enumerate(blocks[0 : stop_layer + 1]):
            attn_heads = blk.attn.get_attention_map() 
            grad = blk.attn.get_attn_gradients()
            input = blk.get_input().squeeze()
            attn = blk.attn.get_output().squeeze()
            resid_1 = blk.get_input_plus_attn().squeeze()
            mlp = blk.get_mlp_val().squeeze()
            
            # Use the head importance to pick the most important head
            grad_temp = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            attn_temp = attn_heads.reshape(-1, attn_heads.shape[-2], attn_heads.shape[-1])
            Ih = torch.mean(torch.matmul(attn_temp.transpose(-1,-2), grad_temp).abs(), dim=(-1,-2))
            Ih = Ih / torch.sum(Ih)
            max_heads = torch.max(attn_heads * Ih.reshape(1, num_heads, 1, 1), dim = 1)[0].detach()

            # Multiply the attention of each layer by its bottom up gradient.
            # "Bottom up" gradient for layer 4 is found by classifying the output of layer 4 
            # and then taking the gradient to the input.
            # This is in contrast to taking the gradient from the last layer (output) to layer 4.
            if withgrad == True:
                block_classifcation_probs = prob[i]
                gradient = torch.autograd.grad(torch.unbind(block_classifcation_probs[:, target_class]), attn_heads, retain_graph=True)[0][0]
                max_heads = (gradient.mean(dim = 0, keepdim=True) * max_heads).clamp(0)

            all_layer_attentions.append(max_heads)

            # take two norm of the model components
            norm = 2
            # first residual connection is between the input and attention output
            # gather the norms of each and normalize their ratio
            one_norm = torch.stack((torch.linalg.norm(input, ord = norm, dim = 1), torch.linalg.norm(attn, ord = norm, dim = 1)))
            one_norm = torch.nn.functional.normalize(one_norm, p = 1, dim = 0)
            all_layer_biases_resid_1.append(one_norm)

            # second residual connection is between the first residual (input + attention out) and the MLP output
            # gather the norms of each and normalize their ratio
            one_norm = torch.stack((torch.linalg.norm(resid_1, ord = norm, dim = 1), torch.linalg.norm(mlp, ord = norm, dim = 1)))
            one_norm = torch.nn.functional.normalize(one_norm, p = 1, dim = 0)
            all_layer_biases_resid_2.append(one_norm)

        InFlow, _ = compute_InFlow(all_layer_attentions, all_layer_biases_resid_1, all_layer_biases_resid_2, ablate)

        # select some target patch and all of the image patches
        InFlow = InFlow[:, target_token, 1:]

        patches_per_side = int(np.sqrt(InFlow.shape[-1]))

        return InFlow.reshape(-1, patches_per_side, patches_per_side), (torch.stack(all_layer_biases_resid_1), torch.stack(all_layer_biases_resid_2))

    # https://github.com/XianrenYty/Transition_Attention_Maps
    def generate_transition_attention_maps(self, input, target_class, start_layer = 0, steps = 20, with_integral = True, first_state = False, device = "cuda:0"):
        input.requires_grad = True
        output = self.model(input.to(device), register_hook=True)
        score = output[0][target_class].sum()
        score.backward()

        b, h, s, _ = self.model.blocks[-1].attn.get_attention_map().shape

        num_blocks = len(self.model.blocks)

        # states is CLS token row
        states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        for i in range(start_layer, num_blocks)[::-1]:
            attn = self.model.blocks[i].attn.get_attention_map().mean(1)

            states_ = states
            # states column vector MVM w self-attn 
            states = torch.einsum('biw, bwh->h', states, attn).reshape(b, 1, s)

            # add residual
            states += states_

        total_gradients = torch.zeros(b, h, s, s).to(device)
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backprop
            output = self.model(data_scaled, register_hook=True)
            score = output[0][target_class].sum()
            score.backward()

            # call grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients

        if with_integral:
            W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        else:
            W_state = self.model.blocks[-1].attn.get_attn_gradients().clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        
        if first_state:
            states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        
        final = states * W_state
        input.requires_grad = False
        
        patches_per_side = int(np.sqrt(s - 1))

        return states[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side), W_state[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side), final[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side), self.model.blocks[-1].attn.get_attention_map().mean(1)[0, 0, 1:], gradients
    
    # https://github.com/jiaminchen-1031/transformerinterp/blob/master/ViT/baselines/ViT/ViT_explanation_generator.py
    def bidirectional(self, input, target_class, steps=20, start_layer=4, samples=20, noise=0.2, mae=False, dino=False, ssl=False, device = "cuda:0"):
        input.requires_grad = True
        output = self.model(input.to(device), register_hook=True)
        score = output[0][target_class].sum()
        score.backward()
        
        b, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape

        R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(device)

        for nb, blk in enumerate(self.model.blocks):
            if nb < start_layer - 1:
                continue
            
            grad = blk.attn.get_attn_gradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            cam = blk.attn.get_attention_map()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            Ih = torch.mean(torch.matmul(cam.transpose(-1,-2), grad).abs(), dim=(-1,-2))
            Ih = Ih/torch.sum(Ih)
            cam = torch.matmul(Ih, cam.reshape(num_head,-1)).reshape(num_tokens,num_tokens)
            R = R + torch.matmul(cam.to(device), R.to(device))

        if ssl:
            if mae:
                return R[:, 1:, 1:].abs().mean(axis=1)
            elif dino:
                return (R[:, 1:, 1:].abs().mean(axis=1)+R[:, 0, 1:].abs())
            else:
                return R[:, 0, 1:].abs()
        
        # integrated gradients
        total_gradients = torch.zeros(b, num_head, num_tokens, num_tokens).to(device)
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backprop
            output = self.model(data_scaled, register_hook=True)
            score = output[0][target_class].sum()
            score.backward()

            # calc grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients        
       
        W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
        attr = W_state * R

        input.requires_grad = False

        patches_per_side = int(np.sqrt(num_tokens - 1))

        if mae:
            return attr[:, 1:, 1:].mean(axis=1)
        elif dino:
            return (attr[:, 1:, 1:].mean(axis=1) + attr[:, 0, 1:])
        else:
            return attr[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side), R[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side)