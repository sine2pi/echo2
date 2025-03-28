

```python
# %%
import os
import math
import warnings

import time
import logging
import torch
from torch import nn, Tensor
import numpy as np
from torch.nn import functional as F
from typing import Tuple, Optional, Dict
import gzip
import base64
from datetime import datetime
from contextlib import contextmanager
from torch import einsum
from einops import rearrange, repeat
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.tensorboard.writer import SummaryWriter
import evaluate
from datasets import load_dataset
from transformers import WhisperTokenizer, WhisperFeatureExtractor
import transformers
from tqdm.notebook import tqdm
from dataclasses import dataclass

from torch.amp import autocast
from itertools import chain

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
transformers.utils.logging.set_verbosity_error()
device = torch.device(device="cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
tokenizer = WhisperTokenizer.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small")


# %%
@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    text_layerA: int
    text_layerB: int
    text_act: str
    text_debug: int
    text_checkpoint: bool
    scale_text_embedding: bool
    
    mels: int
    audio_ctx: int
    audio_dims: int
    audio_head: int
    audio_layerA: int
    audio_layerB: int
    audio_act: str
    audio_debug: int
    audio_checkpoint: bool
    scale_audio_embedding: bool
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
  

def create_attention_mask(batch_size, seq_len, is_causal=True, padding_mask=None, device=None):
    """Create a standardized attention mask for all attention mechanisms"""
    if is_causal:
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device), diagonal=1
        ).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    else:
        mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=device).bool()
    
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        mask = mask | (~padding_mask)
    
    return mask

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape
        
    def forward(self, x: Tensor) -> Tensor:
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_normalized = x_float * torch.rsqrt(variance + self.eps)
        return (x_normalized * self.weight).type(x.dtype)
    
    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype))
    
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class ParameterCycler:
    def __init__(self, parameters):
        self.parameters = parameters
        self.current_idx = 0

    def toggle_requires_grad(self):
        for i, param in enumerate(self.parameters):
            param.requires_grad = i == self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self.parameters)

def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
    return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, dims, ctx):
        super(PositionalEncoding, self).__init__()
        self.dims = dims
        self.ctx = ctx
        self.pe = self.get_positional_encoding(max_seq_len=ctx)
        
    def get_positional_encoding(self, max_seq_len):
        pe = torch.zeros(max_seq_len, self.dims)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dims, 2, dtype=torch.float32) * (-math.log(10000.0) / self.dims))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe.to(device)
    
    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        x = x * math.sqrt(self.dims)
        x = x + pe
        return x


# %%

class RotaryEmbedding(nn.Module):

    def __init__( self, dim, theta = 10000, num_freqs = 1, learned_freq = True, theta_rescale_factor = 1., 
                 use_quaternion = False, rot_scale = 1.0, rot_count = 1, use_projection = False, proj_dim = 3, 
                 proj_scale = 0.1, ): 
        super().__init__()
        
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        self.freqs = nn.Parameter(torch.arange(0, num_freqs) * (2 * math.pi / theta), requires_grad=learned_freq)
        
        self.register_buffer('dummy', torch.tensor(0), persistent=False)
        
        self.use_quaternion = use_quaternion
        self.use_projection = use_projection
        self.proj_dim = proj_dim
        self.proj_scale = proj_scale
        
        if use_quaternion:
            self.dparam = nn.Parameter(torch.zeros(1))
            self.rscale = rot_scale
            self.rot = rot_count
            self.tscale = 1.0
            
            pairs = []
            for i in range(0, dim-1, 2):
                pairs.append(torch.tensor([i, i+1]))
            self.pairs = nn.Parameter(torch.stack(pairs), requires_grad=False)
            
            self.thetas = nn.Parameter(torch.ones(len(self.pairs)) * (2 * math.pi / len(self.pairs)), 
                                      requires_grad=False)
            
            if use_projection:
                self.proj_down = None
                self.proj_up = None

    @property
    def device(self):
        return self.dummy.device

    def q_rotation(self, x, theta, u, v=None):

        eps = 1e-8
        u_norm = torch.norm(u, p=2)
        u = u / (u_norm + eps)
        
        w = torch.cos(theta / 2)
        vec = torch.sin(theta / 2) * u
        
        x_shape = x.shape
        x = x.reshape(-1, 3)
        
        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        
        x_rot = x + torch.clamp(2 * (w * uv_cross + uuv_cross), min=-10.0, max=10.0)
        
        return x_rot.reshape(*x_shape)

    def rotation_matrix(self, dims, i, j, theta):
        """Create a rotation matrix for dimensions i,j with angle theta."""
        G = torch.eye(dims, device=theta.device)
        c, s = torch.cos(theta), torch.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = -s, s
        
        if dims == 3:
            u = torch.eye(dims, device=theta.device)[i]
            v = torch.eye(dims, device=theta.device)[j]
            if theta < 0: 
                Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=abs(theta), u=u, v=v)
            else:
                Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=theta, u=u, v=v)
            G = (G + Q) / 2
            
        return G

    def rotations(self, x):

        direction = torch.sigmoid(self.dparam) * 2 - 1
        rotate = int(round(self.rscale * self.rot))
        
        head_dim = x.shape[-1]
        
        for k in range(min(rotate, len(self.pairs))):
            i, j = self.pairs[k].long()
            if i >= head_dim or j >= head_dim:
                continue
    
            theta = direction * self.thetas[k] * self.tscale
            G = self.rotation_matrix(dims=head_dim, i=i.item(), j=j.item(), theta=theta)
            
            x_shape = x.shape
            x = x.reshape(-1, head_dim)
            x = x @ G
            x = x.reshape(*x_shape)
        
        return x

    def _ensure_projectiolayerAs(self, x):
      
        if self.proj_down is None or self.proj_down.weight.device != x.device:
            head_dim = x.shape[-1] 
            
            self.proj_down = nn.Linear(head_dim, self.proj_dim, bias=False).to(x.device)
            self.proj_up = nn.Linear(self.proj_dim, head_dim, bias=False).to(x.device)
            
            with torch.no_grad():
                nn.init.orthogonal_(self.proj_down.weight, gain=self.proj_scale)
                nn.init.orthogonal_(self.proj_up.weight, gain=self.proj_scale)
                
                U, S, V = torch.svd(self.proj_down.weight)
                S_inv = 1.0 / (S + 1e-6) 
                S_inv = torch.clamp(S_inv, max=10.0)
                pseudo_inv = V @ torch.diag(S_inv) @ U.t()
                self.proj_up.weight.copy_(pseudo_inv * self.proj_scale)

    def project_and_rotate(self, x):

        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        
        with torch.no_grad():
            x_norm = torch.norm(x_flat, dim=1, keepdim=True)
            if torch.max(x_norm) > 1e3:
                x_flat = x_flat * (1e3 / torch.max(x_norm))
        
        if x.shape[-1] > 3 and self.use_projection:
            self._ensure_projectiolayerAs(x)
            x_3d = self.proj_down(x_flat)
            if torch.isnan(x_3d).any():
                return x.reshape(*orig_shape)
            x_3d_rot = self.rotations(x_3d)
            if torch.isnan(x_3d_rot).any():
                x_rot = self.proj_up(x_3d)
            else:
                x_rot = self.proj_up(x_3d_rot)
            alpha = 0.9
            x_rot = alpha * x_rot + (1-alpha) * x_flat
            
            if torch.isnan(x_rot).any():
                return x.reshape(*orig_shape)
        else:
            x_rot = self.rotations(x_flat)
        return x_rot.reshape(*orig_shape)

    def apply_rotary(self, freqs, t, start_index=0, scale=1., seq_dim=-2, freqs_seq_dim=None):
        dtype = t.dtype
        
        def _exists(val):
            return val is not None
        
        def _slice_at_dim(tensor, dim_slice, dim):
            dim += (tensor.ndim if dim < 0 else 0)
            colons = [slice(None)] * tensor.ndim
            colons[dim] = dim_slice
            return tensor[tuple(colons)]
        
        def _rotate_half(x):
            x = rearrange(x, '... (d r) -> ... d r', r=2)
            x1, x2 = x.unbind(dim=-1)
            x = torch.stack((-x2, x1), dim=-1)
            return rearrange(x, '... d r -> ... (d r)')
        
        if not _exists(freqs_seq_dim):
            if freqs.ndim == 2 or t.ndim == 3:
                freqs_seq_dim = 0
                
        if t.ndim == 3 or _exists(freqs_seq_dim):
            ctx = t.shape[seq_dim]
            freqs = _slice_at_dim(freqs, slice(-ctx, None), dim=freqs_seq_dim)
            
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]
        
        t_transformed = (t_middle * freqs.cos() * scale) + (_rotate_half(t_middle) * freqs.sin() * scale)
        out = torch.cat((t_left, t_transformed, t_right), dim=-1)
        
        return out.type(dtype)

    def rotate_qk(self, t, seq_dim=None, offset=0, scale=None):

        if self.use_quaternion:
            if self.use_projection and t.shape[-1] > 3:
                return self.project_and_rotate(t)
            else:
                return self.rotations(t)
        else:
            ctx = t.shape[2]
            device, dtype = t.device, t.dtype
            
            seq = torch.arange(ctx, device=device, dtype=dtype) + offset
            
            freqs = self.forward(seq)
            
            scale = scale if scale is not None else 1.0
            return self.apply_rotary(freqs, t, scale=scale, seq_dim=2)
    
    def learned_rotations(self, rotations, t, start_index = 0, freq_ranges = None):
        if exists(freq_ranges):
            rotations = einsum('..., f -> ... f', rotations, freq_ranges)
            rotations = rearrange(rotations, '... r f -> ... (r f)')

        rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
        return self.apply_rotary(rotations, t, start_index = start_index)

    def forward(self, t):

        freqs = self.freqs
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        return freqs


# %%

@contextmanager
def disable_sdpa():
    prev_state = MultiheadA.use_sdpa
    try:
        MultiheadA.use_sdpa = False
        yield
    finally:
        MultiheadA.use_sdpa = prev_state

class MultiheadA(nn.Module):
    use_sdpa = True

    def __init__(self, dims: int, head: int, max_dist=512):
        super().__init__()
        self.head = head
        self.head_dim = dims // head
        self.max_dist = max_dist
        self.scale = self.head_dim ** -0.5
        self.query = Linear(dims, dims)
        self.key = Linear(dims, dims, bias=False)
        self.value = Linear(dims, dims)
        self.out = Linear(dims, dims)

        self.rotary = RotaryEmbedding(
            dim=dims//head,
            use_quaternion=False,
            use_projection=False,
            rot_scale=1.0,
            rot_count=2
        )

    def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
        """Reshape tensor to [batch, head, ctx, head_dim] with contiguous memory."""
        return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        batch_size, seq_len = x.shape[:2]
        
        q = self.query(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self._attention(q, k, v, mask)
        return self.out(wv), qk

    def _attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx, dims = q.shape
        scale = (dims // self.head) ** -0.5

        q = self._shape(q, ctx, batch)
        k = self._shape(k, k.size(1), batch)
        v = self._shape(v, v.size(1), batch)

        is_causal = False
        if mask is not None:
            if mask.dim() == 4 and mask.dtype == torch.bool:
                is_causal = True
            elif mask.dim() <= 3:
                mask = create_attention_mask(
                    batch_size=batch,
                    seq_len=ctx,
                    is_causal=True,
                    device=q.device
                )
                is_causal = True

        if MultiheadA.use_sdpa:
            try:
                a = scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=mask if mask is not None and mask.dim() == 4 else None,
                    is_causal=is_causal,
                    scale=scale
                )
                out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
                return out, None
            except RuntimeError:
                pass
                
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        if mask is not None:
            if mask.dim() == 4:
                mask_q_len = min(mask.size(2), q.size(2))
                mask_k_len = min(mask.size(3), k.size(2))
                attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len].masked_fill(
                    mask[:, :, :mask_q_len, :mask_k_len], float("-inf")
                )
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, ctx, dims)
        
        return out, attn
    


# %%
class ProjectionModule(nn.Module):
    """Unified projection module that handles query, key, and value transformations."""
    def __init__(self, dims: int, head: int, proj_type: str = "query", use_bias: bool = True):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.proj_type = proj_type
        self.scale = self.head_dim ** -0.25 if proj_type != "value" else 1.0
        self.proj = Linear(in_features=dims, out_features=dims, bias=use_bias)
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(tensor=self.proj.weight, std=0.02)
        if hasattr(self.proj, 'bias') and self.proj.bias is not None:
            nn.init.zeros_(tensor=self.proj.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        batch, ctx = x.shape[:2]
        proj = self.proj(x)
        
        proj = proj.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        if self.proj_type in ["query", "key"]:
            proj = proj * self.scale
        return proj

def calculate_attention(q, k, v, mask=None, temperature=1.0, use_sdpa=True):
    """Attention calculation with zero-padding for invalid tokens."""
    if use_sdpa:
        try:
            if mask is not None:
                if mask.dtype == torch.bool:
                    float_mask = torch.zeros_like(mask, dtype=torch.float)
                    float_mask = float_mask.masked_fill(mask, float('-inf'))
                    attn_output = scaled_dot_product_attention(
                        q, k, v, attn_mask=float_mask)
                else:
                    attn_output = scaled_dot_product_attention(
                        q, k, v, attn_mask=mask)
            else:
                attn_output = scaled_dot_product_attention(
                    q, k, v, attn_mask=None)
            return attn_output, None
        except RuntimeError:
            pass
    scale = 1.0 / temperature if temperature > 0 else 1.0
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    if mask is not None:
        if mask.dim() == 4:
            q_len, k_len = q.size(2), k.size(2)
            mask_q_len = min(mask.size(2), q_len)
            mask_k_len = min(mask.size(3), k_len)
            
            if mask.dtype == torch.bool:
                mask_part = mask[:, :, :mask_q_len, :mask_k_len]
                attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len].masked_fill(
                    mask_part, float("-inf")
                )
            else:
                attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len] + mask[:, :, :mask_q_len, :mask_k_len]
    attn = F.softmax(attn, dim=-1)
    
    if mask is not None and mask.dtype == torch.bool:
        binary_mask = (~mask).float()
        attn = attn * binary_mask
        attn_sum = attn.sum(dim=-1, keepdim=True)
        attn = attn / (attn_sum + 1e-6)
    attn_output = torch.matmul(attn, v)
    return attn_output, attn

class BaseAttention(nn.Module):
    """Base class for attention mechanisms with common functionality."""
    use_sdpa = True
    
    def __init__(self, dims: int, head: int, max_dist: int = 512):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_dist = max_dist
        self.scale = self.head_dim ** -0.25
        
    def _reshape_to_output(self, attn_output, batch, ctx):
        """Reshape attention output to original dimensions."""
        return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims)

class AttentionCombiner(BaseAttention):
    """Combines separate Q and KV representations for attention computation."""
    def __init__(self, dims: int, head: int):
        super().__init__(dims, head)
        self.out = Linear(in_features=dims, out_features=dims)
        nn.init.normal_(tensor=self.out.weight, std=0.02)
        nn.init.zeros_(tensor=self.out.bias)

    @autocast('cuda', enabled=True)
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Processes and combines attention inputs."""
        if q.dim() == 3:
            batch, ctx, _ = q.shape
            q = q.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(batch, k.size(1), self.head, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(batch, v.size(1), self.head, self.head_dim).permute(0, 2, 1, 3)
        else:
            batch = q.size(0)
            ctx = q.size(2)
        attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa)
        output = self._reshape_to_output(attn_output, batch, ctx)
        return self.out(output)

class AdaptiveUpdateAttention(BaseAttention):
    """Attention implementation with content-dependent update frequencies."""
    def __init__(self, dims: int, head: int, max_dist=512):
        super().__init__(dims, head, max_dist)
        
        self.query_module = ProjectionModule(dims, head, "query")
        self.key_module = ProjectionModule(dims, head, "key")
        self.value_module = ProjectionModule(dims, head, "value")
        self.combiner = AttentionCombiner(dims, head)
        self.key_update_predictor = nn.Sequential(
            Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())
        self.value_update_predictor = nn.Sequential(
            Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())

        self.update_threshold = 0.5
        self.stored_key_cache = None
        self.stored_value_cache = None

    def should_update_key(self, x: torch.Tensor) -> torch.Tensor:
        """Predict whether the key should be updated based on content."""
        avg_rep = x.mean(dim=1)
        return self.key_update_predictor(avg_rep) > self.update_threshold

    def should_update_value(self, x: torch.Tensor) -> torch.Tensor:
        """Predict whether the value should be updated based on content."""
        avg_rep = x.mean(dim=1)
        return self.value_update_predictor(avg_rep) > self.update_threshold

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        """Process inputs with adaptive update mechanism."""
        batch, ctx, _ = x.shape
        
        q = self.query_module(x)
        
        kv_input = xa if xa is not None else x
        device = kv_input.device

        if kv_cache is None:
            k = self.key_module(kv_input)
            v = self.value_module(kv_input)
            
            self.stored_key_cache = k
            self.stored_value_cache = v
        else:
            update_k = self.should_update_key(kv_input)
            update_v = self.should_update_value(kv_input)
            
            if update_k.any():
                new_k = self.key_module(kv_input)
                if self.stored_key_cache is not None:
                    update_mask = update_k.view(-1, 1, 1, 1).expand_as(self.stored_key_cache)
                    k = torch.where(update_mask, new_k, self.stored_key_cache)
                else:
                    k = new_k
            else:
                k = self.stored_key_cache if self.stored_key_cache is not None else self.key_module(kv_input)
            
            if update_v.any():
                new_v = self.value_module(kv_input)
                if self.stored_value_cache is not None:
                    update_mask = update_v.view(-1, 1, 1, 1).expand_as(self.stored_value_cache)
                    v = torch.where(update_mask, new_v, self.stored_value_cache)
                else:
                    v = new_v
            else:
                v = self.stored_value_cache if self.stored_value_cache is not None else self.value_module(kv_input)
            
            self.stored_key_cache = k
            self.stored_value_cache = v
        
        output = self.combiner(q, k, v, mask=mask)
        
        return output

class Refiner:
    """Q-learning based refiner for adaptive attention span."""
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.R = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_value = 0.0

    def get_value(self, state, action):
        return self.R.get((state, action), self.default_value)

    def set_value(self, state, action, value):
        self.R[(state, action)] = value

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        else:
            action_values = [self.get_value(state, a) for a in range(self.actions)]
            return np.argmax(action_values)

    def update(self, state, action, reward, next_state):
        next_values = [self.get_value(next_state, a) for a in range(self.actions)]
        best_next_value = max(next_values)

        old_value = self.get_value(state, action)
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - old_value
        new_value = old_value + self.alpha * td_error
        self.set_value(state, action, new_value)

class Predictor(nn.Module):
    """Neural predictor for span scale estimation."""
    def __init__(self, dims):
        super().__init__()
        self.linear = Linear(in_features=dims, out_features=1)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, global_out):
        if global_out.dim() > 2:
            global_out = global_out.mean(dim=1)
        scale = torch.sigmoid(self.linear(global_out))
        return scale

class AdaptiveSpan(BaseAttention):
    """Attention with adaptive span size."""
    def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
        super().__init__(dims, head, max_dist)
        self.sharpen = sharpen
        self.temp_scale = temp_scale
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    @autocast('cuda', enabled=True)
    def forward(self, query, key, value, max_dist=None, max_span=None, span_scale=None):
        if max_dist is None:
            max_dist = self.max_dist
        if max_span is None:
            max_span = query.shape[1]
        if span_scale is None:
            span_scale = self.span_scale
            
        span_mean = span_scale.mean().item()
        span_len = min(int(max_span * span_mean), query.shape[1], key.shape[1], value.shape[1])
        eff_span = min(span_len, max_dist)
        
        if eff_span == 0:
            batch = query.shape[0]
            return (torch.zeros(batch, eff_span, self.dims, device=query.device), None)
            
        q_span = query[:, :eff_span, :]
        k_span = key[:, :eff_span, :]
        v_span = value[:, :eff_span, :]

        batch = q_span.shape[0]

        reshape_dims = (batch, -1, self.head, self.head_dim)
        q = q_span.view(*reshape_dims).permute(0, 2, 1, 3)
        k = k_span.view(*reshape_dims).permute(0, 2, 1, 3)
        v = v_span.view(*reshape_dims).permute(0, 2, 1, 3)

        temperature = (
            1.0 + self.temp_scale * (1.0 - span_mean)
            if self.sharpen
            else 0.5 + self.temp_scale * span_mean
        )
        
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            attn_output, weights = calculate_attention(
                q, k, v, None, temperature, BaseAttention.use_sdpa
            )
            out = self._reshape_to_output(attn_output, batch, eff_span)

        return out, weights

class MyelinatedLayer(BaseAttention):
    def __init__(self, dims, head, layerAs=6, sparsity_threshold=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layerAs = layerAs
        self.sparsity_threshold = sparsity_threshold
        
        self.shared_head = AdaptiveSpan(dims, head)
        
        self.node_predictors = nn.ModuleList([
            nn.Sequential(
                LayerNorm(dims),
                Linear(dims, 1),
                nn.Sigmoid()
            ) for _ in range(layerAs)
        ])
        
        for i in range(layerAs):
            self.layers.append(nn.ModuleDict({
                'ln': LayerNorm(dims),
                'gate': nn.Sequential(Linear(dims, 1), nn.Sigmoid()),
                'adapter': Linear(dims, dims) if i % 2 == 0 else None
            }))
        
        self.policy_net = nn.Sequential(
            Linear(dims, 128),
            nn.ReLU(),
            Linear(128, 3)
        )
        
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
        
        n_mlp = dims * 4
        self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(Linear(dims, n_mlp), nn.GELU(), Linear(n_mlp, dims))
        self.mlp_ln = LayerNorm(dims)
        
        self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
        self.memory_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        
    def shared_head(self, norm_x, mask=None, kv_cache=None):
        batch_size, seq_len = norm_x.shape[:2]
        
        q = norm_x.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
        k = norm_x.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
        v = norm_x.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
        
        attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return attn_output


    def predict_node_importance(self, x, layer_idx):
        """Dynamically determine if processing should occur at this node"""
        importance = self.node_predictors[layer_idx](x)
        return (importance > self.sparsity_threshold).float()
    
    def forward(self, x, xa=None, mask=None, kv_cache=None):
        batch_size, seq_len = x.shape[:2]
        
        working_memory = self.working_memory.expand(batch_size, -1, -1)
        
        original_x = x
        
        pooled_representation = x.mean(dim=1)
        policy_logits = self.policy_net(pooled_representation)
        policy = F.softmax(policy_logits, dim=-1)
        
        jump_history = []
        i = 0
        while i < self.layerAs:
            layer = self.layers[i]
            
            node_importance = self.predict_node_importance(x, i)
            
            if node_importance.mean() < 0.2 and i > 0:
                i += 1
                jump_history.append(i)
                continue
                
            norm_x = layer['ln'](x)
            
            attn_mask = mask
            if mask is None:
                attn_mask = node_importance.squeeze(-1).unsqueeze(1).expand(-1, seq_len, -1)
            else:
                attn_mask = mask * node_importance.squeeze(-1).unsqueeze(1).expand(-1, seq_len, -1)
                
            if node_importance.mean() > 0.3:
                attn_output = self.shared_head(norm_x, mask=attn_mask, kv_cache=kv_cache)[0]
                
                if layer['adapter'] is not None:
                    attn_output = layer['adapter'](attn_output)
                
                gate_value = layer['gate'](norm_x).unsqueeze(-1)
                x = x + gate_value * attn_output
                
                memory_gate = self.memory_gate(x)
                working_memory = memory_gate * working_memory + (1 - memory_gate) * x.mean(dim=1, keepdim=True)
            
            jump_prob = policy[:, 1] if i < self.layerAs - 1 else torch.zeros_like(policy[:, 1])
            should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
            
            if should_jump:
                jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
                
                i_next = min(i + jump_length, self.layerAs - 1)
                skip_weight = self.jump_weights[min(jump_length-1, 2)]
                
                x = x + skip_weight * original_x + (1-skip_weight) * working_memory
                
                i = i_next
                jump_history.append(i)
            else:
                i += 1
        
        mlp_importance = self.mlp_gate(x)
        mlp_output = self.mlp(self.mlp_ln(x))
        x = x + mlp_importance * mlp_output
        
        return x, {'jumps': jump_history}


class IntegratedAttention(nn.Module):
    """Combines local adaptive span and global content-dependent attention with RL-based adaptation."""
    def __init__(self, ctx, dims, head, max_dist=512, win_size=256, max_span=384, temp_scale=0.01):
        super().__init__()
        self.head = head
        self.max_dist = max_dist
        self.ctx = ctx
        self.dims = dims
        self.max_span = max_span
        self.sliding_window = win_size
        self.temp_scale = temp_scale
        self.sharpen = True
        self.head_dim = dims // head
        self.batch = None

        self.refiner = Refiner(
            states=10000, actions=10, alpha=0.1, gamma=0.9, epsilon=0.1
        )
        
        self.span_pred = Predictor(dims=dims)
        self.attn_local = AdaptiveSpan(
            dims=dims, head=head, max_dist=max_dist, sharpen=True, temp_scale=temp_scale
        )
        self.attn_global = AdaptiveUpdateAttention(dims=dims, head=head, max_dist=max_dist)
        
        self.projection = Linear(in_features=2 * dims, out_features=dims)
        self.ln_a = LayerNorm(normalized_shape=dims)
        self.ln_b = LayerNorm(normalized_shape=dims)

        mask = torch.empty(max_span, max_span).fill_(float("-inf")).triu_(diagonal=1)
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("window_mask", None, persistent=False)
        self.register_buffer("threshold", torch.tensor(1e-4), persistent=False)
        self.register_buffer("s_factor", torch.tensor(0.1), persistent=False)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        """Process input with integrated local and global attention."""
        batch_size, seq_len = x.shape[:2]
        
        if mask is None or mask.dim() != 4:
            mask = create_attention_mask(
                batch_size=batch_size, 
                seq_len=seq_len,
                is_causal=True,
                device=x.device
            )
        
            
        local = self.ln_a(x)
        globe = self.ln_b(x)

        globe_out = self.attn_global(globe, globe, globe)
        freq_scale = self.span_pred(globe_out)
        state = self.extract(local)
        action = self.refiner.choose_action(state=state)
        refine = self.action_scale(action=action)
        span_scale = torch.clamp(freq_scale * refine, min=0.0, max=1.0)
        span_mean = span_scale.mean().item()

        with torch.no_grad():
            current_win_size = max(1, int(self.sliding_window * span_mean))
            current_span_len = max(1, int(self.max_span * span_mean))

            effective_max = min(self.max_dist, local.size(1))
            local_max = min(self.max_dist, current_span_len, current_win_size)
            globe_max = effective_max

        self.attn_local.max_dist = local_max
        self.attn_global.max_dist = globe_max

        local_out = self.slide_win(
            x=local,
            win_size=current_win_size,
            span_len=current_span_len,
            span_scale=span_scale,
            mask=mask,
        )
        
        with torch.no_grad():
            quality = self.quality(output=local_out)
            next_state = self.extract(local_out)
            self.refiner.update(
                state=state, action=action, reward=quality, next_state=next_state)
        
        combined = torch.cat([local_out, globe_out], dim=-1)
        x = self.projection(combined)
        return x

    def quality(self, output):
        """Calculate quality metric for reinforcement learning."""
        with torch.no_grad():
            safe_output = output.clamp(min=1e-10)
            entropy = -(safe_output * torch.log(safe_output)).sum(-1).mean()
            coverage = (output > 0.01).float().mean()
            return float(coverage - 0.1 * entropy)

    def extract(self, x):
        """Extract state features for RL agent."""
        with torch.no_grad():
            meadims = x.mean(dim=(0, 1))
            var_state = x.var(dim=(0, 1), unbiased=False)
            state = torch.cat([meadims, var_state])
            state_id = self.discretize(state.cpu().numpy())
        return state_id

    def discretize(self, state):
        """Convert continuous state to discrete state ID."""
        bins = np.linspace(-1, 1, num=10)
        state_discrete = np.digitize(state, bins)
        state_hash = hash(tuple(state_discrete))
        state_id = state_hash % (self.refiner.states - 1)
        return state_id

    def action_scale(self, action):
        """Convert discrete action to continuous scale factor."""
        span_value = action / (self.refiner.actions - 1)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        span_scale = torch.tensor([span_value], device=device, dtype=dtype)
        return span_scale
        
    @autocast('cuda', enabled=True)
    def _focus(self, query, key, value, span_scale, mask=None):
        """Iterative attention refinement with zero-padding for invalid tokens."""
        max_iterations = 10
        iteration = 0
        prev_attn = torch.zeros_like(input=query)
        attn_out = torch.zeros_like(input=query)
        attn_weights = None

        threshold = self.threshold.item()
        s_factor = self.s_factor.item()

        while iteration < max_iterations:
            span_len = int(self.max_span * span_scale.mean().item())
            span_len = min(span_len, query.size(1), key.size(1), value.size(1))
            eff_span = min(span_len, self.max_dist)

            if eff_span == 0:
                break

            q_span = query[:, :eff_span, :]
            k_span = key[:, :eff_span, :]
            v_span = value[:, :eff_span, :]

            batch, ctx, dims = q_span.size()
            
            q = q_span.view(batch, ctx, self.head, -1).transpose(1, 2)
            k = k_span.view(batch, ctx, self.head, -1).transpose(1, 2)
            v = v_span.view(batch, ctx, self.head, -1).transpose(1, 2)

            if self.sharpen:
                temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
            else:
                temperature = 0.5 + self.temp_scale * span_scale.mean().item()
            
            scale = (dims // self.head) ** -0.5
            attn = torch.matmul(q, k.transpose(-1, -2)) * scale
            
            if mask is not None:
                if mask.dim() == 4:
                    q_len, k_len = q.size(2), k.size(2)
                    mask_q_len = min(mask.size(2), q_len)
                    mask_k_len = min(mask.size(3), k_len)
                    
                    effective_mask = torch.zeros_like(attn)
                    
                    effective_mask[:, :, :mask_q_len, :mask_k_len] = mask[:, :, :mask_q_len, :mask_k_len].float() * float("-inf")
                    
                    attn = attn + effective_mask
            
            attn = F.softmax(attn, dim=-1)
            
            if mask is not None and mask.dim() == 4:
                binary_mask = (mask == 0).float()
                
                attn = attn * binary_mask
                
                attn_sum = attn.sum(dim=-1, keepdim=True)
                attn = attn / (attn_sum + 1e-6)
                
            attn_output = torch.matmul(attn, v)
            attn_out = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)

            diff = torch.abs(attn_out - prev_attn).mean()
            dynamic_threshold = threshold + s_factor * diff

            if diff < dynamic_threshold:
                break

            prev_attn = attn_out
            query = query + attn_out
            iteration += 1
            
        return attn_out, attn_weights

    @autocast('cuda', enabled=True)
    def slide_win(self, x, win_size, span_len, span_scale, mask=None):
        """Process input with sliding window attention."""
        batch, ctx, dims = x.size()
        self.batch = batch
        num_windows = (ctx + win_size - 1) // win_size
        output = torch.zeros_like(x)
        device = x.device

        for i in range(num_windows):
            start_idx = i * win_size
            end_idx = min((i + 1) * win_size, ctx)
            window_size = end_idx - start_idx

            key_start = max(0, start_idx - span_len + win_size)
            key_end = min(start_idx + span_len, ctx)
            span_size = key_end - key_start

            query = x[:, start_idx:end_idx, :]
            key = x[:, key_start:key_end, :]
            value = key

            window_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    window_mask = mask[:, :, start_idx:end_idx, key_start:key_end]
                    
                    if window_mask.size(1) == 1:
                        window_mask = window_mask.expand(-1, self.head, -1, -1)

            attn_out, _ = self._focus(
                query=query,
                key=key,
                value=value,
                span_scale=span_scale,
                mask=window_mask,
            )

            output[:, start_idx:end_idx, :] = attn_out

        return output


# %%

class Residual(nn.Module):
    def __init__(self, dims: int, head: int, act: str, debug=False, cross_attention=False):
        if debug is True:
            print(f"Residual check:{dims} {head} {act}")

        super().__init__()
        self.dims = dims
        self.head = head
        self.cross_attention = cross_attention

        act_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }

        self.act = act_map.get(act, nn.GELU())
        
        self.attna = MultiheadA(dims=dims, head=head)
        self.attnc = (MultiheadA(dims=dims, head=head) if cross_attention else None)
    
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), self.act, Linear(mlp, dims))

        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims)
        self.lnc = RMSNorm(dims) if cross_attention else None 

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attna(self.lna(x), mask=mask, kv_cache=kv_cache)[0]
        if self.attnc:
            x = x + self.attnc(self.lnc(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.lnb(x))
        return x
    

class AudioEncoder(nn.Module):
    def __init__(self, mels: int, ctx: int, dims: int, head: int, layerA: int, layerB: int, checkpoint: bool, act: str,  
                 scale_embedding, debug=None):
        if debug == 1:
            print(
                f"AudioEncoder check: {mels} {ctx} {dims} {head} {checkpoint} {act} {layerA} {layerB}"
            )
        super().__init__()
        self.ctx = ctx
        self.dims = dims
        self.head = head
        self.layerA = layerA
        self.layerB = layerB
        self.checkpoint = checkpoint

        self.embed_scale = math.sqrt(dims) if scale_embedding else 1.0

        act_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU()
        }
        self.act = act_map.get(act, nn.GELU())

        self.conv1 = Conv1d(mels, dims, kernel_size=3, padding=1)
        self.conv2 = Conv1d(dims, dims, kernel_size=3, stride=2, padding=1)

        self.register_buffer("positional_embedding", sinusoids(ctx, dims))

        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, act=act) 
                                                for _ in range(layerA)]) if layerA > 0 else None)

        self.blockB = (nn.ModuleList([IntegratedAttention(ctx=ctx, dims=dims, head=head)
                                                for _ in range(layerB)]) if layerB > 0 else None)
        
        self.ln_enc = RMSNorm(dims)
        self.expected_ctxgth = ctx * self.conv1.stride[0] * self.conv2.stride[0]

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x)) 
        x = x.permute(0, 2, 1)
        x = nn.functional.dropout(x, p=0.001)
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        x = x * self.embed_scale
        for block in chain(self.blockA or [], self.blockB or []):
            x = block(x, mask=mask)
            if isinstance(x, tuple):
                x = x[0]
            else:
                x = x
        x = self.ln_enc(x)
        return x

class TextDecoder(nn.Module):
    def __init__(self, vocab: int, ctx: int, dims: int, head: int, layerA: int, layerB: int, checkpoint: bool, act: str, debug=None):
        if debug == 2: 
            print(f"TextDecoder check: {vocab} {ctx} {dims} {head} {checkpoint} {act} {layerA} {layerB}")
        super().__init__()
        self.checkpoint = checkpoint
        self.ctx = ctx
        self.dims = dims
        self.head = head
        self.layerA = layerA
        self.layerB = layerB
        self.act = act
        self.pad_token_id = 0

        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims))
        self.positional_encoding = PositionalEncoding(ctx=ctx, dims=dims)

        self.ln_dec = RMSNorm(dims)

        self.blockA = (nn.ModuleList([Residual(dims=dims, head=head, act=act, cross_attention=False) 
                                      for _ in range(layerA)]) if layerA > 0 else None)

        self.blockB = (nn.ModuleList([IntegratedAttention(ctx=ctx, dims=dims, head=head)
                                      for _ in range(layerB)]) if layerB > 0 else None)

        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None) -> Tensor:
        batch_size = x.size(0)
        self.ctx = x.size(1)
        
        mask = create_attention_mask(
            batch_size=batch_size,
            seq_len=self.ctx,
            is_causal=True,
            padding_mask=(x != self.pad_token_id),
            device=x.device
        )
        
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset: offset + x.shape[-1]])
        x = self.positional_encoding(x)
        x = x.to(xa.dtype)

        for block in chain(self.blockA or [], self.blockB or []):
            x = block(x, xa, mask=mask, kv_cache=kv_cache)

        x = self.ln_dec(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits


# %%

class Echo(nn.Module):
    
    def __init__(self, param: Dimensions, debug=None):
        super().__init__()

        self.debug = debug
        self.param = param

        self.encoder = AudioEncoder(
            mels=param.mels,
            ctx=param.audio_ctx,
            dims=param.audio_dims,
            head=param.audio_head,
            layerA=param.audio_layerA,
            layerB=param.audio_layerB,
            checkpoint=param.audio_checkpoint,
            act=param.audio_act,
            scale_embedding=param.scale_audio_embedding,
            debug=param.audio_debug,
        )
        self.decoder = TextDecoder(
            vocab=param.vocab,
            ctx=param.text_ctx,
            dims=param.text_dims,
            head=param.text_head,
            layerA=param.text_layerA,
            layerB=param.text_layerB,
            checkpoint=param.text_checkpoint,
            act=param.text_act,
            debug=param.text_debug,
        )

        all_head = torch.zeros(
            self.param.text_layerA, self.param.text_head, dtype=torch.bool
        )
        all_head[self.param.text_layerA // 2 :] = True
        self.register_buffer("alignment_head", all_head.to_sparse(), persistent=False)

    def to(self, device):
        super().to(device)
        self.encoder.to(device) 
        self.decoder.to(device)
        return self
    
    def set_alignment_head(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.param.text_layerA, self.param.text_head
        )
        self.register_buffer("alignment_head", mask.to_sparse(), persistent=False)

    def embed_audio(self, input_features: torch.Tensor):
        return self.encoder(input_features)

    def logits(self,input_ids: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(input_ids, audio_features)

    def forward(self, input_features: torch.Tensor, input_ids=None, labels=None, decoder_inputs_embeds=None) -> Dict[str, torch.Tensor]:

        if input_ids is None and decoder_inputs_embeds is None:
            if labels is not None:
                input_ids = shift_tokens_right(
                    labels, self.param.pad_token_id, self.param.decoder_start_token_id)
            else:
                raise ValueError("You have to provide either decoder_input_ids or labels")
        
        encoded_audio = self.encoder(input_features)
        logits = self.decoder(input_ids, encoded_audio)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        
        return {"logits": logits, "loss": loss, "labels": labels, "input_ids": input_ids}

    
    @property
    def is_multilingual(self):
        return self.param.vocab >= 51865

    @property
    def num_languages(self):
        return self.param.vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.param.text_ctx:
    
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def save_adaptive_output(module, _, output):
            if isinstance(output, tuple) and len(output) == 2:
                tensor_output, cache_updates = output
                
                module_key = f"{module}_key"
                module_value = f"{module}_value"
                
                if module_key not in cache or tensor_output.shape[1] > self.param.text_ctx:
                    cache[module_key] = cache_updates["key_cache"]
                    cache[module_value] = cache_updates["value_cache"]
                else:
                    cache[module_key] = torch.cat([cache[module_key], cache_updates["key_cache"]], dim=1).detach()
                    cache[module_value] = torch.cat([cache[module_value], cache_updates["value_cache"]], dim=1).detach()

                return tensor_output
            return output

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiheadA):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))
            elif isinstance(layer, AdaptiveUpdateAttention):
                hooks.append(layer.register_forward_hook(save_adaptive_output))

            self.encoder.apply(install_hooks)
        self.decoder.apply(install_hooks)
        
        return cache, hooks
    
    def adjust_freq(self, loss, factor=1.0025) -> float | int:
            if self.adjust_counter % 25 == 0:
                if loss < self.best_loss:
                    new_freq=self.freq*factor
                else:
                    new_freq=self.freq/factor
                self.update_freq(new_freq=new_freq)
                self.freq=new_freq
                self.best_loss=loss
            self.adjust_counter += 1
            return self.freq
            
    def update_freq(self, new_freq):
        self.new_freq=new_freq
        for name, module in self.encoder.named_modules():
            if isinstance(module, (RotaryEmbedding)):
                module.update_freq(new_freq=self.new_freq)

    def generate(self, mel: torch.Tensor, max_length: int = 512) -> torch.Tensor:
        audio_features = self.encoder(mel)
        return self.decoder.generate(audio_features, max_length=max_length)
        
    def attach_debug_hooks(self):
        """Attach hooks to debug attention scores."""
        def debug_attention_scores(module, input, output):
            print(f"Debugging {module.__class__.__name__}: Attention scores shape: {output[0].shape}")

        for name, module in self.encoder.named_modules():
            if isinstance(module, MultiheadA) or isinstance(module, IntegratedAttention):
                module.register_forward_hook(debug_attention_scores)

        for name, module in self.decoder.named_modules():
            if isinstance(module, MultiheadA) or isinstance(module, IntegratedAttention):
                module.register_forward_hook(debug_attention_scores)



# %%


def ctx_to_samples(audio_ctx, hop_length):
    samples_token = hop_length * 2
    n_samples = audio_ctx * samples_token
    return n_samples

def load_wave(wave_data, sample_rate):
    if isinstance(wave_data, str):
        waveform, sr = torchaudio.load(uri=wave_data, normalize=False)
    elif isinstance(wave_data, dict):
        waveform = torch.tensor(data=wave_data["array"]).float()
        sr = wave_data["sampling_rate"]
    else:
        raise TypeError("Invalid wave_data format.")
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    if sr != sample_rate:
        original_length = waveform.shape[1]
        target_length = int(original_length * (sample_rate / sr))
        
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
        
        if abs(waveform.shape[1] - target_length) > 1:
            new_waveform = torch.zeros((waveform.shape[0], target_length), dtype=waveform.dtype, device=waveform.device)
            copy_length = min(waveform.shape[1], target_length)
            new_waveform[:, :copy_length] = waveform[:, :copy_length]
            waveform = new_waveform
    
    return waveform.flatten()

def pad(array, target_length, axis=-1, dtype: torch.dtype = torch.float32):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(dtype)
    if torch.is_tensor(array):
        if array.shape[axis] > target_length:
            array = array.index_select(
                dim=axis,
                index=torch.arange(
                    end=target_length, device=array.device, dtype=torch.long
                ),
            )
        if array.shape[axis] < target_length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, target_length - array.shape[axis])
            array = F.pad(
                input=array, pad=[pad for sizes in pad_widths[::-1] for pad in sizes]
            )
        array = array.to(dtype=dtype)
    else:
        raise TypeError(
            f"Unsupported input type: {type(array)}. Expected torch.Tensor or np.ndarray."
        )
    return array

def exact_div(x, y):
    assert x % y == 0
    return x // y

def process_audio(audio, audio_ctx, mels, hop_length, n_fft, sr):

    audio = load_wave(wave_data=audio, sample_rate=sr)
    n_samples = ctx_to_samples(audio_ctx=audio_ctx, hop_length=hop_length)
    audio = pad(array=audio, target_length=n_samples)

    transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mels,
    )
    
    mel_spectrogram = transform(audio)

    target_frames = exact_div(n_samples, hop_length) 
    mel_spectrogram = pad(array=mel_spectrogram, target_length=target_frames, axis=-1)

    log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    
    return log_mel

tokenizer = WhisperTokenizer.from_pretrained(
    pretrained_model_name_or_path="openai/whisper-small"
)

class DataCollator:
    def __init__(self, tokenizer, audio_ctx, text_ctx, mels, n_fft, hop_length, sample_rate=16000, device="cpu"):
        self.tokenizer = tokenizer
        self.text_ctx = text_ctx
        self.audio_ctx = audio_ctx
        self.sample_rate = sample_rate
        self.mels = mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.decoder_start_token_id = 50258
        self.pad_token_id = 50257
        self.eos_token_id = 50257

    def __call__(self, features):
        batch = len(features)

        max_time_frames = 0
        max_text_length = 0
        
        processed_features = []
        for feature in features:
            audio = process_audio(
                audio=feature["audio"],
                audio_ctx=self.audio_ctx,
                mels=self.mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                sr=self.sample_rate,
            )
            time_frames = audio.shape[-1]
            max_time_frames = max(max_time_frames, time_frames)
            
            transcript = feature["transcription"]
            encoded_input = self.tokenizer.encode(transcript, add_special_tokens=False)
            encoded_label = self.tokenizer.encode(transcript, add_special_tokens=False)
            
            decoder_input = [self.decoder_start_token_id] + encoded_input
            labels = encoded_label + [self.eos_token_id]
            
            max_text_length = max(max_text_length, len(decoder_input), len(labels))
            
            processed_features.append({
                "audio": audio,
                "decoder_input": decoder_input,
                "labels": labels
            })
        
        max_text_length = min(max_text_length, self.text_ctx)
        
        batch_audio = torch.zeros(
            size=(batch, self.mels, max_time_frames),
            dtype=torch.float32,
            device=self.device,
        )
        batch_input_ids = torch.full(
            size=(batch, max_text_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        batch_labels = torch.full(
            size=(batch, max_text_length),
            fill_value=-100,
            dtype=torch.long,
            device=self.device,
        )

        for i, feature in enumerate(processed_features):
            audio = feature["audio"]
            time_frames = audio.shape[-1]
            decoder_input = feature["decoder_input"][:max_text_length]
            labels = feature["labels"][:max_text_length]
            
            batch_audio[i, :, :time_frames] = torch.tensor(data=audio, dtype=torch.float32)
            batch_input_ids[i, :len(decoder_input)] = torch.tensor(data=decoder_input, dtype=torch.long)
            batch_labels[i, :len(labels)] = torch.tensor(data=labels, dtype=torch.long)

        return {
            "input_features": batch_audio,
            "input_ids": batch_input_ids,
            "labels": batch_labels,
        }

metric = evaluate.load(path="wer")

def compute_metrics(pred, tokenizer):
    pred_ids = pred["predictions"]
    label_ids = pred["label_ids"]
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    else:
        pred_ids = pred_ids
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "pred_str": pred_str, "label_str": label_str}


def generate_predictions(model, input_features_encoded, tokenizer, device, batch_size, min_length):
    decoder_start_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 50258
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 50257
    
    generated_ids = torch.full(size=(batch_size, 1), fill_value=decoder_start_token_id, dtype=torch.long, device=device)
    
    max_length = 150
    for i in range(max_length - 1):
        with torch.no_grad():
            curr_output = model.decoder(generated_ids, input_features_encoded)
        next_token_logits = curr_output[:, -1, :]
        
        if i < min_length:
            next_token_logits[:, eos_token_id] = float('-inf')
        next_tokens = torch.argmax(input=next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat(tensors=[generated_ids, next_tokens], dim=1)
        if (next_tokens == eos_token_id).all() and i >= min_length:
            break
    return generated_ids


# %%


def train_and_evaluate(model, tokenizer, train_loader, eval_loader, optimizer, scheduler, loss_fn, max_steps=10000, device='cuda', 
    accumulation_steps=1, clear_cache=True, log_interval=10, eval_interval=100, save_interval=1000, 
    warmup_steps=0, checkpoint_dir="checkpoint_dir", log_dir="log_dir"):
    
    model.to(device)
    global_step = 0
    scaler = torch.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    train_iterator = iter(train_loader)
    total_loss = 0
    step_in_report = 0
    dataset_epochs = 0

    if warmup_steps > 0:
        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
            
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        logging.info(f"Using learning rate warmup for {warmup_steps} steps")
    else:
        warmup_scheduler = None

    progress_bar = tqdm(total=max_steps, desc="Training Progress", leave=True, colour='green')

    model.train()
    optimizer.zero_grad()

    while global_step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            dataset_epochs += 1
            print(f"Starting dataset epoch {dataset_epochs}")

            if step_in_report > 0:
                avg_loss = total_loss / step_in_report
                logging.info(f"Dataset iteration complete - Steps: {global_step}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0
                step_in_report = 0

        start_time = time.time()

        input_features = batch['input_features'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].long().to(device)

        with torch.autocast(device_type="cuda"):
            input_features_encoded = model.encoder(input_features)
            decoder_output = model.decoder(input_ids, input_features_encoded)
            logits = decoder_output.view(-1, decoder_output.size(-1))
            active_logits = logits.view(-1, decoder_output.size(-1))
            active_labels = labels.view(-1)
            active_mask = active_labels != -100
            active_logits = active_logits[active_mask]
            active_labels = active_labels[active_mask]
            loss = loss_fn(active_logits, active_labels)

        total_loss += loss.item()
        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (global_step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if clear_cache:
                torch.cuda.empty_cache()

        end_time = time.time()
        samples_per_sec = len(batch['input_features']) / (end_time - start_time)

        if global_step % log_interval == 0:
            if warmup_steps > 0 and global_step < warmup_steps:
                lr = optimizer.param_groups[0]['lr']
            else:
                lr = scheduler.get_last_lr()[0]
                
            writer.add_scalar(tag='Loss/train', scalar_value=total_loss / (global_step + 1), global_step=global_step)
            writer.add_scalar(tag='LearningRate', scalar_value=lr, global_step=global_step)
            writer.add_scalar(tag='SamplesPerSec', scalar_value=samples_per_sec, global_step=global_step)
            logging.info(f"Step {global_step} - Loss: {total_loss / (global_step + 1):.4f}, LR: {lr:.8f}")
        
        if global_step % eval_interval == 0:
            model.eval()
            eval_start_time = time.time()
            eval_loss = 0
            all_predictions = []
            all_labels = []
            batch_count = 0
            total_samples = 0


            with torch.no_grad():
                for eval_batch in eval_loader:
                    input_features = eval_batch['input_features'].to(device)
                    input_ids = eval_batch['input_ids'].to(device)
                    labels = eval_batch['labels'].long().to(device)

                    batch = input_features.size(0)
                    total_samples += batch

                    input_features_encoded = model.encoder(input_features)
                    decoder_output = model.decoder(input_ids, input_features_encoded)
                    logits = decoder_output.view(-1, decoder_output.size(-1))
                    loss = loss_fn(logits, labels.view(-1))
                    eval_loss += loss.item()
                    all_predictions.extend(torch.argmax(decoder_output, dim=-1).cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())

                    batch_count += 1

            if warmup_steps > 0 and global_step < warmup_steps:
                lr = optimizer.param_groups[0]['lr']
            else:
                lr = scheduler.get_last_lr()[0]
                
            eval_time = time.time() - eval_start_time
            loss_avg = eval_loss / batch_count if batch_count > 0 else 0
            predictions = {"predictions": np.array(all_predictions, dtype=object), "label_ids": np.array(all_labels, dtype=object)}
            metrics = compute_metrics(pred=predictions, tokenizer=tokenizer)

            writer.add_scalar('Loss/eval', loss_avg, global_step)
            writer.add_scalar('WER', metrics['wer'], global_step)
            writer.add_scalar('EvalSamples', total_samples, global_step)
            writer.add_scalar('EvalTimeSeconds', eval_time, global_step)
            
            print(f"STEP {global_step} • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}")
            print(f"PRED: '{metrics['pred_str'][0]}'")
            print(f"REF : '{metrics['label_str'][0]}'")
            print()

            logging.info(f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}")
            model.train()

        if global_step % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved at step {global_step} to {checkpoint_path}")
            
        global_step += 1
        step_in_report += 1

        if warmup_steps > 0 and global_step <= warmup_steps:
            warmup_scheduler.step()
        else:
            scheduler.step()

        avg_loss = total_loss / (global_step + 1)
        if warmup_steps > 0 and global_step <= warmup_steps:
            lr = optimizer.param_groups[0]['lr']
        else:
            lr = scheduler.get_last_lr()[0]
            
        postfix_dict = {'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.6f}', 'WER': f'{metrics["wer"]:.4f}' if 'wer' in metrics else 'N/A', 'samp/sec': f'{samples_per_sec:.1f}'}
        progress_bar.set_postfix(postfix_dict, refresh=True)
        progress_bar.update(1)

    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed after {global_step} steps. Final model saved to {final_model_path}")
    writer.close()
    progress_bar.close()


# %%

if __name__ == "__main__":

    checkpoint_dir = './output/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join("./output/logs/rottestresidual", datetime.now().strftime(format="%m-%d_%H"))
    os.makedirs(name=log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'), filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    token = ""
    dataset = load_dataset(
        path="google/fleurs",
        name="en_us",
        streaming=False,
        token=token,
        trust_remote_code=True,
        cache_dir="E:/cache",
    ).select_columns(column_names=["audio", "transcription"])


    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", feature_size=80, sampling_rate=16000)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")

    dataset = load_dataset(
        path="mozilla-foundation/common_voice_17_0",
        name="en",
        streaming=True,
        token=token,
        trust_remote_code=True,
        cache_dir="E:/cache",
    ).select_columns(column_names=["audio", "sentence"]).rename_column("sentence", "transcription")
   
    debug = None
    
    param = Dimensions(
        mels=80,
        audio_ctx=1500,
        audio_head=4,
        audio_layerA=4,
        audio_layerB=0,
        audio_dims=512,
        audio_act="gelu",
        audio_checkpoint=False,
        scale_audio_embedding=False,
        audio_debug=debug,
        vocab=len(tokenizer),
        text_ctx=448,
        text_head=4,
        text_layerA=4,
        text_layerB=0,
        text_dims=512,
        text_act="gelu",
        text_checkpoint=False,
        scale_text_embedding=False,
        text_debug=debug,
        decoder_start_token_id = 50258,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id
        )
    
    model = Echo(param=param).to('cuda')
    DataCollator = DataCollator(tokenizer=tokenizer, 
                                audio_ctx=param.audio_ctx, 
                                text_ctx=param.text_ctx, 
                                mels=param.mels, 
                                n_fft=400,
                                hop_length=160,
                                sample_rate=16000,
                                device='cuda')



    train_dataloader = DataLoader(
        dataset=dataset["train"],
        batch_size=1, 
        collate_fn=DataCollator,
        num_workers=0)

    eval_dataloader = DataLoader(
        dataset=dataset["test"].take(100),
        batch_size=1,
        collate_fn=DataCollator,
        num_workers=0)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0001, last_epoch=-1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
    train_and_evaluate(model=model, 
        tokenizer=tokenizer, 
        train_loader=train_dataloader, 
        eval_loader=eval_dataloader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        loss_fn=loss_fn, 
        warmup_steps=100,
        max_steps=1100,
        device='cuda', 
        accumulation_steps=1, 
        clear_cache=False, 
        log_interval=100,
        eval_interval=100, 
        save_interval=1000,
        checkpoint_dir=checkpoint_dir, 
        log_dir=log_dir
        )


from tensorboard import program
log_dir = "./output/logs" 
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()
print(f"TensorBoard started at {url}")







```


