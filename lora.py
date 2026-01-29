import torch

import torch.nn as nn

from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
import copy


    
class LowRankConv1d(nn.Module):
    def __init__(self, in_features, out_features, rank, zeros=False):
        super().__init__()
        self.zeros = zeros
        if not zeros: 
            self.lora_A = torch.nn.Conv1d(in_features,
                             rank,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            self.lora_B = torch.nn.Conv1d(rank,
                             out_features,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        
            nn.init.normal_(self.lora_A.weight, std=1/rank)
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        if self.zeros:
            return torch.zeros_like(x)
        return self.lora_B( self.lora_A( x ))


class loraAttentionBlock(nn.Module):   
    def __init__(self, base_layer, rank=4, qkv=[True, True, True], alpha=1, dropout=0):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.qkv = qkv
        self.scaling = alpha / rank


        # Dimensions from the base layer
        self.channels = base_layer.channels

        # Initialize LoRA layers using LowRankConv (since input is image b,c,h,w)
        
        self.lora_q = LowRankConv1d(self.channels, self.channels, self.rank, zeros=not qkv[0])
        self.lora_k = LowRankConv1d(self.channels, self.channels, self.rank, zeros=not qkv[1])
        self.lora_v = LowRankConv1d(self.channels, self.channels, self.rank, zeros=not qkv[2])

        self.lora_proj_out = LowRankConv1d(self.channels, self.channels, self.rank)


    def forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)

        x_norm = self.base.norm(x_flat)
        base_qkv = self.base.qkv(x_norm)

        lora_qkv = torch.cat([self.lora_q(x_flat), self.lora_k(x_flat), self.lora_v(x_flat)], dim=1)*self.scaling
        h = self.base.attention(base_qkv + lora_qkv)
        h_final = self.base.proj_out(h) + self.lora_proj_out(h)*self.scaling
        desired_output = (x_flat + h_final).reshape(b, c, *spatial)
        return desired_output


def apply_lora_to_layer(layer, rank=4, alpha=1, qkv=[True, False, True]):
    '''
    applies vera to a layer if it can
    '''
    new_layer = copy.deepcopy(layer)
    if isinstance(layer, AttentionBlock):
        new_layer = loraAttentionBlock(layer, rank=rank, qkv=qkv, alpha=alpha)

    return new_layer


#removed LinearAttention because not ready I am tired voglio morire
available_targets = [AttentionBlock]


def ignorant_lora(model, target_class=available_targets, rank=4, qkv=[True, False, True], alpha=1):
    """
    Applies Lora wherever it can in the model
    """
    #for name, module in model.named_modules():
    #    if isinstance(module, AttentionBlock):
    #        print(f"mashallah")

    #print([ name for name, module in model.named_modules() if any(isinstance(module, target) for target in target_class)])
    layers_to_replace = []
    for name, module in model.named_modules():
        for target in target_class: 
            if isinstance(module, target):
                layers_to_replace.append((name, module))

    for full_name, old_layer in layers_to_replace:

        if '.' in full_name:
            parent_name, child_name = full_name.rsplit('.', 1)

            parent_module = model.get_submodule(parent_name)
        else:
            parent_name = ""
            child_name = full_name
            parent_module = model

        #print(f"Replacing layer: {full_name}")
        new_layer = apply_lora_to_layer(old_layer, rank=rank, qkv=qkv, alpha=alpha)
        #print("New Parameters theoretical:", sum([len(n) for n, p in new_layer.named_parameters() if 'lora' in n]))
        setattr(parent_module, child_name, new_layer)
    return model


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")



class loraModel(nn.Module):
    def __init__(self, base_model, rank = 4, alpha=1, qkv=[True, False, True], targets=available_targets):
        super().__init__()
        
        self.model = ignorant_lora(base_model, rank=rank, alpha=alpha, qkv=qkv, target_class=targets)

    def forward(self, x, t=None, **kwargs):
        return self.model(x, t, **kwargs)

    def __getattr__(self, name):
            try:
                # 1. First, let PyTorch find standard attributes (like self.model, parameters, etc.)
                return super().__getattr__(name)
            except AttributeError:
                # 2. If PyTorch doesn't have it, ONLY THEN check the wrapped model
                return getattr(self.model, name)

    def set_trainable_parameters(self):
        '''
        Sets parameters of the adapters and of the head to requires_grad=True, all other parameters to False
        '''
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        print_trainable_parameters(self.model)


    def state_dict(self, destination=None, prefix='', keep_vars=False, print_percentage=False):
        """
        override of the classical state_dict method, only shows trainable parameters
        """
        full_state_dict = self.model.state_dict(destination, prefix, keep_vars)
        
        filtered_dict = {
            k: v for k, v in full_state_dict.items() 
            if f"{k}" in [n for n, p in self.model.named_parameters() if p.requires_grad]
        }
        
        if print_percentage:
            print_trainable_parameters(self.model)
        return filtered_dict


