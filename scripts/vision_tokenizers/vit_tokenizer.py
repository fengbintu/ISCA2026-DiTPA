import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from timm.models.vision_transformer import (
    PatchDropout,
    PatchEmbed,
    VisionTransformer,
    checkpoint_seq,
    resample_abs_pos_embed,
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from vision_tokenizers.clip import clip_vit_hf
from vision_tokenizers.qformer import QFormer, qformer_base_hf
from vision_tokenizers.token_learner import TokenLearnerModule
from transformers import CLIPModel, CLIPProcessor
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def forward_depth_wrist_ref(self: VisionTransformer, image, remove_index=None, return_attn=False):
    img_patch_embed = self.patch_embed(image)  
    image_x = self._pos_embed(img_patch_embed) 
    
    attn_weights_list = []
    for i in range(12): # vit backbone 
        ### Self-attention block ###
        img_copy = image_x
        img_norm1 = self.blocks[i].norm1(image_x)

        B, N, C = img_norm1.shape 
        qkv = self.blocks[i].attn.qkv(img_norm1).reshape(B, N, 3, self.blocks[i].attn.num_heads, self.blocks[i].attn.head_dim).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv.unbind(0) 
        scale = 1.0/(q.shape[-1]**0.5) 
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  

        attn_weights = attn_scores.softmax(dim=-1) 

        attn_weights_list.append(attn_weights.detach())

        att_v = torch.matmul(attn_weights, v) 
        att_v = att_v.transpose(1, 2).reshape(B, N, C)

        img_attn = self.blocks[i].attn.proj(att_v)
        img_ls1 = self.blocks[i].ls1(img_attn) 
        img_inter = img_copy + img_ls1 

        ### MLP block ###
        img_copy = img_inter
        img_norm2 = self.blocks[i].norm2(img_inter) 
        
        img_fc1 = self.blocks[i].mlp.fc1(img_norm2)  
        img_act = self.blocks[i].mlp.act(img_fc1)  
        img_fc2 = self.blocks[i].mlp.fc2(img_act)  

        img_ls2 = self.blocks[i].ls2(img_fc2) 
        image_x = img_copy + img_ls2 
    image_x = self.norm(image_x)  
    
    x = image_x[:, self.num_prefix_tokens:] 
    
    if return_attn:
        return x, attn_weights_list
    else:
        return x


class RT1ViTImageTokenizer(nn.Module):
    def __init__(
        self,
        embedding_output_dim: int,
        use_qformer: bool = True,
        qformer_depth: int = 4,
        num_tokens: int = 32,
        dropout_rate=0.1,
        freeze_backbone=True,
        use_wrist_img=False,
        use_depth_img=False,
        input_size=None,
        vit_forward_version = None,
    ):
        super().__init__()
        self.embedding_output_dim = embedding_output_dim # 768
        img_size = ast.literal_eval(input_size) # [224,224]
        
        self.tokenizer = create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, img_size=img_size, drop_path_rate=0.1, proj_drop_rate=0.1
        )
        self.tokenizer.forward_depth_wrist_ref = forward_depth_wrist_ref.__get__(self.tokenizer, VisionTransformer)

        self.tokenizer.use_wrist_img = use_wrist_img
        self.tokenizer.use_depth_img = use_depth_img

        if freeze_backbone:
            for param in self.tokenizer.parameters():
                param.requires_grad = False

        self.use_qformer = use_qformer # True
        self.qformer = QFormer(
            num_queries=num_tokens,  # 32
            embed_dim=self.embedding_output_dim,
            depth=qformer_depth,
            num_heads=self.embedding_output_dim // 64,
            mlp_ratio=4,
            qkv_bias=False,
            norm_layer=nn.LayerNorm,
            dropout_rate=dropout_rate,
            drop_path=0.0,
            use_checkpoint=not freeze_backbone,
            with_film=True,
            cross_dim=self.tokenizer.embed_dim,
        )
        self.num_tokens = num_tokens # 32

    @property
    def tokens_per_context_image(self) -> int:
        return self.num_tokens

    def forward(self, image, context=None, wrist_image=None, depth_image=None):
        b, t, c, h, w = image.shape      
        image = image.flatten(0, 1) 

        if context is not None:
            context = context.flatten(0, 1) 
            while context.dim() != 2:
                context = context.mean(dim=-2)  
        
        tokens, attention_weights_vit = self.get_image_embeddings(image, wrist_image, depth_image, return_attn=True)  

        if self.use_qformer:
            tokens, attention_weights = self.qformer(tokens, context, return_attn=True)

        token_num = tokens.shape[1] 
        tokens = tokens.view(b, t, token_num, -1)
        new_tokens = tokens 

        return new_tokens

    def get_image_embeddings(self, image: torch.Tensor, wrist_image=None, depth_image=None, enable_vit_compare=False, return_attn=False) -> torch.Tensor:   

        image_tokens_ref = self.tokenizer.forward_depth_wrist_ref(image, return_attn=return_attn) 
        
        return image_tokens_ref 
