from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn as nn

from ldm.modules.diffusionmodules.util import checkpoint
import copy



class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        

        nn.init.normal_(self.lora_down.weight, std=1/rank)
        nn.init.zeros_(self.lora_up.weight)

        #print('Number of theoretical', len(self.lora_down.weight.flatten()) + len(self.lora_up.weight.flatten()))
        #print('Number of parameters in Linear: ', len(self.lora_down.weight.flatten()) + len(self.lora_up.weight.flatten()))

    def forward(self, x):

        return self.lora_up(self.lora_down(x))


class LowRankConv(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.lora_A = torch.nn.Conv2d(in_features,
                             rank,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.lora_B = torch.nn.Conv2d(rank,
                             out_features,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        
        nn.init.normal_(self.lora_A.weight, std=1/rank)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B( self.lora_A( x ))





class loraLinearAttention(nn.Module):
    def __init__(self, base_layer, rank=4, qkv=[True, True, True], alpha=1, dropout=0):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.qkv = qkv
        self.scaling = alpha / rank
        self.heads = base_layer.heads
        self.scale = base_layer.scale

        # Dimensions from the base layer
        # to_qkv is a Conv2d(dim, hidden_dim*3, ...)
        self.dim = base_layer.to_qkv.in_channels
        self.hidden_dim = base_layer.to_qkv.out_channels // 3
        
        # Initialize LoRA layers using LowRankConv (since input is image b,c,h,w)
        if qkv[0]:
            self.lora_q = LowRankConv(self.dim, self.hidden_dim, self.rank)
        if qkv[1]:
            self.lora_k = LowRankConv(self.dim, self.hidden_dim, self.rank)
        if qkv[2]:
            self.lora_v = LowRankConv(self.dim, self.hidden_dim, self.rank)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Get original QKV from the frozen base layer
        # The base layer creates a fused tensor, so we chunk it into 3
        qkv = self.base.to_qkv(x).chunk(3, dim=1)
        q_orig, k_orig, v_orig = qkv

        # 2. Inject LoRA updates (Add to Q, K, V separately)
        if self.qkv[0]:
            q = q_orig + self.lora_q(x) * self.scaling
        else:
            q = q_orig

        if self.qkv[1]:
            k = k_orig + self.lora_k(x) * self.scaling
        else:
            k = k_orig

        if self.qkv[2]:
            v = v_orig + self.lora_v(x) * self.scaling
        else:
            v = v_orig

        # 3. Re-run the specific Linear Attention logic (copied from attention.py)
        
        # Rearrange for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), (q, k, v))

        # Softmax Normalization (Specific to Linear Attention)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        # Efficient computation using einsum
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        # Reshape back to image format
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        # 4. Final projection (using the frozen base output layer)
        return self.base.to_out(out)
    


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class loraSpatialSelfAttention(nn.Module):
    def __init__(self, base_layer, rank=4, qkv=[True, True, False], alpha=1, dropout=0):
        '''
        support_layers must contain SSA_A_x and SSA_B_x for x=qkv
        MUST be Conv2d with in_channels in and out
        '''
        super().__init__()
        self.rank = rank
        self.base = base_layer
        self.in_channels = base_layer.in_channels
        self.qkv = qkv

        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)

        self.norm = Normalize(self.in_channels)

        if qkv[0]:
            self.lora_q = LowRankConv(self.in_channels, self.in_channels, self.rank)
        if qkv[1]:
            self.lora_k = LowRankConv(self.in_channels, self.in_channels, self.rank)
        if qkv[2]:
            self.lora_v = LowRankConv(self.in_channels, self.in_channels, self.rank)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)

        q = self.base.q(h_)
        if self.qkv[0]:
            q += self.lora_q(h_)*self.scaling
        k = self.base.k(h_)
        if self.qkv[1]:
            k += self.lora_k(h_)*self.scaling
        v = self.base.v(h_)
        if self.qkv[2]:
            v += self.lora_v(h_)*self.scaling

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
    

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


class loraCrossAttention(nn.Module):
    def __init__(self, base_layer, rank=4, qkv=[True, True, False], alpha=1, dropout=0):
        super().__init__()
        self.rank = rank
        self.base = base_layer
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)
        
        self.heads = base_layer.heads
        # --- THE FIX: Copy the scale from the base layer ---
        self.scale = base_layer.scale 

        self.qkv = qkv

        self.query_dim = base_layer.to_q.in_features
        self.inner_dim = base_layer.to_q.out_features
        self.context_dim = base_layer.to_k.in_features

        if qkv[0]:
            self.lora_q = LowRankLinear(self.query_dim, self.inner_dim, self.rank)
        if qkv[1]:
            self.lora_k = LowRankLinear(self.context_dim, self.inner_dim, self.rank)
        if qkv[2]:
            self.lora_v = LowRankLinear(self.context_dim, self.inner_dim, self.rank)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.base.to_q(x)
        if self.qkv[0]:
            q += self.lora_q(x) * self.scaling

        context = default(context, x)

        k = self.base.to_k(context)
        if self.qkv[1]:
            k += self.lora_k(context) * self.scaling

        v = self.base.to_v(context)
        if self.qkv[2]:
            v += self.lora_v(context) * self.scaling

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Now self.scale is defined, so this won't crash
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.base.to_out(out) # Ensure to call base.to_out (not self.to_out which doesn't exist)

from  ldm.modules.diffusionmodules.openaimodel import QKVAttention, AttentionPool2d
from ldm.modules.attention import LinearAttention, SpatialSelfAttention, CrossAttention


class loraAttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        base_layer, rank, qkv, 
    ):
        super().__init__()
        self.base_layer = base_layer

        self.embed_dim = base_layer.embed_dim
        self.output_dim = base_layer.output_dim
        self.num_heads = base_layer.num_heads
        self.num_heads_channels = base_layer.num_heads_channels

        self.qkv_proj = nn.Conv1d(self.embed_dim, 3 * self.embed_dim, 1)
        self.c_proj = nn.Conv1d(self.embed_dim, self.output_dim or self.embed_dim, 1)
        self.num_heads = self.embed_dim // self.num_heads_channels
        self.attention = QKVAttention(self.num_heads)

        self.embed_dim = base_layer.qkv_proj.in_channels
        self.output_dim = base_layer.c_proj.out_channels
        self.num_heads = base_layer.num_heads


        if qkv[0]:
            self.lora_q = LowRankConv(self.embed_dim, self.embed_dim, self.rank)
        else: 
            self.lora_q = None
        if qkv[1]:
            self.lora_k = LowRankConv(self.embed_dim, self.embed_dim, self.rank)
        else:
            self.lora_k = None
        if qkv[2]:              
            self.lora_v = LowRankConv(self.embed_dim, self.embed_dim, self.rank)
        else:
            self.lora_v = None

        self.lora_c = LowRankConv(self.embed_dim, self.output_dim, self.rank)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.base_layer.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)

        base_hidden = self.qkv_proj(x)
        lora_hidden = torch.cat(self.lora_q(x), self.lora_k(x), self.lora_v(x), dim=1)

        x = self.attention(base_hidden+lora_hidden)
        x = self.c_proj(x)+self.lora_c(x)
        return x[:, :, 0]
    



def apply_lora_to_layer(layer, rank=4, qkv=[True, False, True]):
    '''
    applies vera to a layer if it can
    '''
    new_layer = copy.deepcopy(layer)
    if isinstance(layer, LinearAttention):
        new_layer = loraLinearAttention(layer, rank=rank, qkv=qkv)
    elif isinstance(layer, SpatialSelfAttention):
        new_layer = loraSpatialSelfAttention(layer, rank=rank, qkv=qkv)
    elif isinstance(layer, CrossAttention):
        new_layer = loraCrossAttention(layer, rank=rank, qkv=qkv)
    elif isinstance(layer, AttentionPool2d):
        new_layer = loraAttentionPool2d(layer, rank=rank, qkv=qkv)

    return new_layer
    


def ignorant_lora(model, target_class):
    """
    Applies Lora wherever it can in the model
    """
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

        print(f"Replacing layer: {full_name}")
        new_layer = apply_lora_to_layer(old_layer)
        #print("New Parameters theoretical:", sum([len(n) for n, p in new_layer.named_parameters() if 'lora' in n]))
        setattr(parent_module, child_name, new_layer)
    return model


#removed LinearAttention because not ready I am tired voglio morire
available_targets = [SpatialSelfAttention, CrossAttention, AttentionPool2d ]

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")



class loraModel(nn.Module):
    def __init__(self, base_model, rank = 4, qkv=[True, False, True], targets=available_targets):
        super().__init__()
        
        self.model = ignorant_lora(base_model, target_class=targets)

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


