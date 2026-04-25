import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
    # print('xformers enabled')
except:
    XFORMERS_IS_AVAILBLE = False
    print("xformers disabled")

from transformers import (
    CLIPTextModel,
    CLIPVisionModel,
    LlamaForCausalLM,
    OPTForCausalLM,
    OPTModel,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.opt.modeling_opt import OPTDecoderLayer


class LlamaXAttention(nn.Module):
    """Memory Efficient Attention layer for Llama, only support causal attn mask"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}" f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # use xformers here
        attn_mask = None if query_states.shape[-2] == 1 else xformers.ops.LowerTriangularMask()
        q, k, v = query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2)
        attn_output = xformers.ops.memory_efficient_attention(q, k.to(dtype=q.dtype), v.to(dtype=q.dtype), attn_bias=attn_mask)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class OPTXAttention(nn.Module):
    """Memory Efficient Attention layer for OPT, only support causal attn mask"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}" f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        assert key_value_states is None and self.is_decoder and layer_head_mask is None and not output_attentions and self.dropout == 0.0

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

        attn_mask = None if query_states.shape[-2] == 1 else xformers.ops.LowerTriangularMask()
        q, k, v = query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2)
        attn_output = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_mask)
        attn_weights_reshaped = None

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class CLIPXAttention(nn.Module):
    """Memory Efficient Attention layer for CLIP, support full & causal attn mask"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        assert (self.dropout == 0.0) and (attention_mask is None)
        attention_mask = xformers.ops.LowerTriangularMask() if causal_attention_mask is not None else None
        q, k, v = query_states, key_states, value_states
        attn_output = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attention_mask)
        attn_weights_reshaped = None

        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


def convert_llama_attn(model: LlamaForCausalLM):
    for layer in model.model.layers:
        layer: LlamaDecoderLayer = layer
        attn_o = layer.self_attn
        attn_x = LlamaXAttention(config=attn_o.config)
        for module_name in ["q_proj", "v_proj", "k_proj", "o_proj"]:
            module_o: nn.Linear = getattr(attn_o, module_name)
            module_x: nn.Linear = getattr(attn_x, module_name)
            module_x.weight.data.copy_(module_o.weight.data)
        layer.self_attn = attn_x
        del attn_o
    print("convert llama self_attn to memory efficient mode successfully")


def convert_opt_attn(model: OPTForCausalLM):
    for layer in model.model.decoder.layers:
        layer: OPTDecoderLayer = layer
        attn_o = layer.self_attn
        bias = attn_o.q_proj.bias is not None
        attn_x = OPTXAttention(
            embed_dim=attn_o.embed_dim, num_heads=attn_o.num_heads, dropout=attn_o.dropout, is_decoder=attn_o.is_decoder, bias=bias
        )
        for module_name in ["q_proj", "v_proj", "k_proj", "out_proj"]:
            module_o: nn.Linear = getattr(attn_o, module_name)
            module_x: nn.Linear = getattr(attn_x, module_name)
            module_x.weight.data.copy_(module_o.weight.data)
        layer.self_attn = attn_x
        del attn_o
    print("convert opt self_attn to memory efficient mode successfully")


def convert_clip_visual_attn(model: CLIPVisionModel):
    for layer in model.vision_model.encoder.layers:
        attn_o = layer.self_attn
        attn_x = CLIPXAttention(config=attn_o.config)
        for module_name in ["q_proj", "v_proj", "k_proj", "out_proj"]:
            module_o: nn.Linear = getattr(attn_o, module_name)
            module_x: nn.Linear = getattr(attn_x, module_name)
            module_x.weight.data.copy_(module_o.weight.data)
            module_x.bias.data.copy_(module_o.bias.data)
        layer.self_attn = attn_x
        del attn_o
    print("convert clip visual self_attn to memory efficient mode successfully")


def convert_clip_text_attn(model: CLIPTextModel):
    for layer in model.text_model.encoder.layers:
        attn_o = layer.self_attn
        attn_x = CLIPXAttention(config=attn_o.config)
        for module_name in ["q_proj", "v_proj", "k_proj", "out_proj"]:
            module_o: nn.Linear = getattr(attn_o, module_name)
            module_x: nn.Linear = getattr(attn_x, module_name)
            module_x.weight.data.copy_(module_o.weight.data)
            module_x.bias.data.copy_(module_o.bias.data)
        layer.self_attn = attn_x
        del attn_o
    print("convert clip text self_attn to memory efficient mode successfully")
