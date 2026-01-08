from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn as nn

from ldm.modules.diffusionmodules.util import checkpoint
import copy




class loraLinearAttention(nn.Module):
    def __init__(self, base_layer, rank=4, qkv = [True, True, False], alpha=1, dropout=0):

        super().__init__()
        self.base = base_layer
        self.rank = rank

        self.dim = base_layer.to_qkv.in_channels
        self.hidden_dim = base_layer.to_out.in_channels

        if qkv[0]:
            self.lora_A_q = torch.nn.Conv2d(self.dim,
                                 self.rank,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
            self.lora_B_q = torch.nn.Conv2d(self.rank,
                                 self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        if qkv[1]:
            self.lora_A_k = torch.nn.Conv2d(self.dim,
                                 self.rank,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
            self.lora_B_k = torch.nn.Conv2d(self.rank,
                                 self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        if qkv[2]:
            self.lora_A_v = torch.nn.Conv2d(self.dim,
                                 self.rank,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
            self.lora_B_v = torch.nn.Conv2d(self.rank,
                                 self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        #TODO includere alpha e dropout

    def forward(self, x):
        original_output = self.base(x)

        #change Forward pass, the original is unclear

        #TODO check dimensions
        partial = self.vera_linatt_middle*self.linatt_A(x)
        vera_output = self.vera_linatt_out*self.linatt_B( partial)
        return original_output+vera_output
    


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

        self.norm = Normalize(self.in_channels)

        if qkv[0]:
            self.lora_A_q = torch.nn.Conv2d(self.in_channels,
                                 self.rank,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
            self.lora_B_q = torch.nn.Conv2d(self.rank,
                                 self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        if qkv[1]:
            self.lora_A_k = torch.nn.Conv2d(self.in_channels,
                                 self.rank,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
            self.lora_B_k = torch.nn.Conv2d(self.rank,
                                 self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        if qkv[2]:
            self.lora_A_v = torch.nn.Conv2d(self.in_channels,
                                 self.rank,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
            self.lora_B_v = torch.nn.Conv2d(self.rank,
                                 self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)

        q = self.base.q(h_)
        if self.qkv[0]:
            q += self.lora_B_q( self.lora_A_q ( h_ ))
        k = self.base.k(h_)
        if self.qkv[1]:
            k += self.lora_B_k( self.lora_A_k ( h_ ))
        v = self.base.v(h_)
        if self.qkv[2]:
            q += self.lora_B_v( self.lora_A_v ( h_ ))

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

        self.qkv = qkv

        self.query_dim = base_layer.to_q.in_features
        self.inner_dim = base_layer.to_q.out_features
        self.context_dim = base_layer.to_k.in_features

        if qkv[0]:
            self.lora_A_q = nn.Parameter(nn.Linear(self.query_dim, self.rank, bias=False))
            self.lora_B_q = nn.Parameter(nn.Linear(self.rank, self.inner_dim, bias=False))
        if qkv[1]:
            self.lora_A_k = nn.Parameter(nn.Linear(self.context_dim, self.rank, bias=False))
            self.lora_B_k = nn.Parameter(nn.Linear(self.rank, self.inner_dim, bias=False))
        if qkv[2]:
            self.lora_A_v = nn.Parameter(nn.Linear(self.context_dim, self.rank, bias=False))
            self.lora_B_v = nn.Parameter(nn.Linear(self.rank, self.inner_dim, bias=False))


    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.base.to_q(x)
        if self.qkv[0]:
            q += self.lora_B_q( self.lora_A_q( x ))

        context = default(context, x)

        k = self.base.to_k(context)
        if self.qkv[1]:
            q += self.lora_B_k( self.lora_A_k( x ))

        v = self.base.to_v(context)
        if self.qkv[2]:
            q += self.lora_B_v( self.lora_A_v( x ))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)



def apply_lora(model, rank=4, qkv=[True, False, True]):
    '''
    applies vera to a transformer block
    '''
    new_model = copy.deepcopy(model)
    new_model.attn1 = loraCrossAttention(new_model.attn1, rank=rank, qkv=qkv)
    new_model.attn2 = loraCrossAttention(new_model.attn2, rank=rank, qkv=qkv)
    return new_model
    


class loraSpatialTransformer(nn.Module):
    def __init__(self, base_transformer, rank = 4, qkv=[True, False, True]):
        super().__init__()
        
        self.model = copy.deepcopy(base_transformer)

        self.model.transformer_blocks = nn.ModuleList(
            [apply_lora(transfblock, rank=rank, qkv=qkv) for transfblock in base_transformer.transformer_blocks]
        )


    def forward(self, x):
        return self.model(x)


    def set_trainable_parameters(self):
        '''
        Sets parameters of the adapters and of the head to requires_grad=True, all other parameters to False
        '''
        for name, param in self.model.named_parameters():
            if "head" in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False



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
            print(f"Keeping {len(filtered_dict)}/{len(full_state_dict)} keys.")
        return filtered_dict



