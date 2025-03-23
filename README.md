
Full model and loop ready to go. 
To add the adapative layers increase layerB. They don't need to have the same number of layers or dimensions. Educational/experimental.  
    
    @dataclass
    class Dimensions:
        vocab: int
        text_ctx: int
        text_dims: int
        text_head: int
        text_layerA: int
        text_layerB: int
        text_act: str
        text_debug: bool
        text_dropout: float
        text_checkpoint: bool
        mels: int
        audio_ctx: int
        audio_dims: int
        audio_head: int
        audio_layerA: int
        audio_layerB: int
        audio_act: str
        audio_debug: bool
        audio_dropout: float
        audio_checkpoint: bool
        scale_embedding: float
    


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
from torch.amp import autocast
from torch.nn import functional as F
from typing import Tuple, Optional, Dict
import gzip
import base64
from datetime import datetime
from itertools import chain
# from separated_attention import QueryModule, KeyModule, ValueModule, AttentionCombiner
# Data handling libraries
import torchaudio
from einops import rearrange
# from torchaudio.transforms import MelSpectrogram
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.tensorboard.writer import SummaryWriter

# ML libraries
import evaluate
from datasets import load_dataset
from transformers import WhisperTokenizer
import transformers
from tqdm.notebook import tqdm
from dataclasses import dataclass

# Set up environment
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
transformers.utils.logging.set_verbosity_error()
device = torch.device(device="cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

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
    text_dropout: float
    text_checkpoint: bool

    mels: int
    audio_ctx: int
    audio_dims: int
    audio_head: int
    audio_layerA: int
    audio_layerB: int
    audio_act: str
    audio_debug: int
    audio_dropout: float
    audio_checkpoint: bool
    scale_embedding: float

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )
def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
    return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()

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


# %%

class rotary(nn.Module):
    def __init__(self, dims, head, freq=10000, debug=False):
        if debug is True:
            print(f"Rotary check: {dims} {head} {freq}")
        super().__init__()
        head_dim = dims // head
        rot = head_dim // 2
        self.dims = dims
        self.head = head
        self.head_dim = head_dim
        self.freq = freq
        self.rot = rot
        self.dparam = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        self.thetas = nn.Parameter(torch.zeros(rot), requires_grad=False)
        self.pairs = nn.Parameter(torch.rand(rot, 2) * head_dim, requires_grad=False)
        self.tscale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rscale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.matrix = nn.Parameter(torch.eye(head_dim), requires_grad=False)
        self.freq_data = 1.0 / (freq ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.invf = nn.Parameter(self.freq_data, requires_grad=False)
        
        self.cycler = ParameterCycler(parameters=[self.dparam, self.matrix, self.invf, self.thetas, self.pairs, self.tscale, self.rscale])

    def vectorize_rotations(self, flat):
        self.batch = flat.size(0)
        G_matrices = []
        for k in range(self.rot):
            i, j = self.pairs[k].long()
            theta = self.thetas[k] * self.tscale
            G = self.rotation_matrix(self.head_dim, i.item(), j.item(), theta)
            G_matrices.append(G)
        G_combined = torch.eye(self.head_dim, device=flat.device)
        for G in G_matrices:
            G_combined = G_combined @ G
        return flat @ G_combined

    def update_freq(self, new_freq):
        if new_freq is not None and new_freq != self.freq:
            self.freq = new_freq
            invf = 1.0 / (self.freq ** (torch.arange(start=0, end=self.hhead_dim, step=2).float() / self.head_dim))
            self.invf.data.copy_(invf)
            self.update_pairs()

    def update_pairs(self):
        pairs = []
        while len(pairs) < self.rot:
            i, j = torch.randint(0, self.h_dim - 1, (2,))
            if i != j and (i, j) not in pairs and (j, i) not in pairs:
                pairs.append((i, j))
        self.pairs.data.copy_(torch.tensor(pairs, dtype=torch.float32))

    def reset_parameters(self):
        nn.init.orthogonal_(self.matrix)
        nn.init.zeros_(self.thetas)

    def q_rotation(self, x, theta, u, v):
        u = u / torch.linalg.matrix_norm(u, ord=2)
        v = v / torch.linalg.matrix_norm(v, ord=2)
        cos_ht = torch.cos(theta / 2)
        sin_ht = torch.sin(theta / 2)
        q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
        q_conj = torch.cat([cos_ht.unsqueeze(0), -sin_ht * u])  # noqa: F841
        x_shape = x.shape
        x = x.view(-1, 3)
        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)
        x_rot = x_rot.view(*x_shape)
        return x_rot

    def rotation_matrix(self, dims, i, j, theta):
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

    def rotations(self, x): # dparam = nn.Parameter(torch.zeros(1))
        direction = torch.sigmoid(self.dparam) * 2 - 1
        rotate = int(torch.round(self.rscale * self.rot))
        for k in range(rotate):
            i, j = self.pairs[k].long()
            theta = direction * self.thetas[k] * self.tscale
            G = self.rotation_matrix(dims=self.head_dim, i=i.item(), j=j.item(), theta=theta)
            x = x @ G
        return x

    @autocast('cuda', enabled = True)
    def forward(self, x):
        self.cycler.toggle_requires_grad()
        x = x.to(device)
        batch, self.ctx, *rest = x.size()

        if len(rest) == 1:
            self.dims = rest[0]
            if self.dims != self.head * self.head_dim:
                raise ValueError(
                    f"Needed {self.head * self.head_dim}, but got too many {self.dims}")
        elif len(rest) == 2:
            self.head, self.head_dim = rest
            if self.head != self.head or self.head_dim != self.head_dim:
                raise ValueError(
                    f"This many head {self.head} and head_dims {self.head_dim} we need, got this many head {self.head} and head_dims {self.head_dim} we did."
)
        else:
            raise ValueError(f"Expected the thingy to be 3D or 4D, but got {x.dim()}D")
        x = rearrange(x, 'b s (h d) -> (b s) h d', h=self.head)
        x = self.vectorize_rotations(x)
        x = x @ self.matrix
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch, h=self.head)
        
        position = torch.arange(end=self.ctx, device=device, dtype=dtype)
        position = rearrange(tensor=position, pattern='s -> s 1')  # [ctx, 1]
        div_term = rearrange(tensor=self.invf, pattern='d -> 1 d')  # [1, dim/2]
        sinusoid = position * div_term  # [ctx, dim/2]

        sin = rearrange(tensor=torch.sin(input=sinusoid), pattern='s d -> 1 s 1 d')  # [1, ctx, 1, dim/2]
        cos = rearrange(tensor=torch.cos(input=sinusoid), pattern='s d -> 1 s 1 d')  # [1, ctx, 1, dim/2]
        
        x = rearrange(tensor=x, pattern='b s (h d) -> b s h d', h=self.head)
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        x_out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x_out = rearrange(x_out, 'b s h d -> b s (h d)')

        x_out = x_out * math.sqrt(self.dims)
        return x_out

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# %%

class MultiheadA(nn.Module):
    use_sdpa = True

    def __init__(self, dims: int, head: int, max_dist: int=256):
        super().__init__()
        self.head = head
        self.max_dist = max_dist
        self.dims = dims
        self.head_dim = dims // head
        self.scale = self.head_dim ** -0.25

        self.query = Linear(dims, dims)
        self.key = Linear(dims, dims, bias=False)
        self.value = Linear(dims, dims)
        self.out = Linear(dims, dims)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        batch, ctx, _ = x.size()
        q = self.query(x)
        
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk
    
    @autocast('cuda', enabled = True)
    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, ctx, dims = q.shape
        scale = (dims // self.head) ** -0.25
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

    
        if  MultiheadA.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:ctx, :ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk

class QueryModule(nn.Module):
    """Dedicated query projection module that handles only query transformations."""
    def __init__(self, dims: int, head: int):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.scale = self.head_dim ** -0.25
        self.q = Linear(in_features=dims, out_features=dims)
        self.init_weights()
    def init_weights(self):
        nn.init.normal_(tensor=self.q.weight, std=0.02)
        if self.q.bias is not None:
            nn.init.zeros_(tensor=self.q.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        batch, ctx = x.shape[:2]
        q = self.q(x)
        
        q = q.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        q = q * self.scale
        return q

class KeyModule(nn.Module):
    """Dedicated key projection module that handles only key transformations."""
    
    def __init__(self, dims: int, head: int):
        """ Args: dims: Input/output dimension size head: Number of attention head"""
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.scale = self.head_dim ** -0.25
        self.key = Linear(in_features=dims, out_features=dims, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(tensor=self.key.weight, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        batch, ctx = x.shape[:2]
        k = self.key(x)
        k = k.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        k = k * self.scale
        return k

class ValueModule(nn.Module):
    """Dedicated value projection module that handles only value transformations."""
    def __init__(self, dims: int, head: int):
    
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.value = Linear(in_features=dims, out_features=dims)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(tensor=self.value.weight, std=0.02)
        if self.value.bias is not None:
            nn.init.zeros_(tensor=self.value.bias)
    
    def forward(self, x: Tensor) -> Tensor:
      
        batch, ctx = x.shape[:2]
        v = self.value(x)
        v = v.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        return v

class KeyValueModule(nn.Module):
    
    def __init__(self, dims: int, head: int):
        super().__init__()
        
        self.key_module = KeyModule(dims, head)
        self.value_module = ValueModule(dims, head)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        k = self.key_module(x)
        v = self.value_module(x)
        return k, v


class AttentionCombiner(nn.Module):
    """Combines separate Q and KV representations for attention computation."""
    use_sdpa = True
    def __init__(self, dims: int, head: int):
        super().__init__()
        
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        
        self.out = Linear(in_features=dims, out_features=dims)
        nn.init.normal_(tensor=self.out.weight, std=0.02)
        nn.init.zeros_(tensor=self.out.bias)

    @autocast('cuda', enabled = True)
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            q, k, v: Tensors of shape [batch, head, ctx, head_dim] or [batch, ctx, dims]
            mask: Optional mask tensor

        """
        if q.dim() == 3:
            batch, ctx, _ = q.shape
            q = q.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(batch, k.size(1), self.head, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(batch, v.size(1), self.head, self.head_dim).permute(0, 2, 1, 3)
        else:
            batch = q.size(0)
            ctx = q.size(2)

        if AttentionCombiner.use_sdpa:
            try:
                attn_output = scaled_dot_product_attention(
                    q, k, v, is_causal=mask is not None and ctx > 1
                )
            except RuntimeError:
                print(f"SDPA failed with shapes: q={q.shape}, k={k.shape}, v={v.shape}")
                attn = torch.matmul(q, k.transpose(-1, -2))
                if mask is not None:
                    if mask.dim() <= 2:
                        mask_to_use = mask[:ctx, :k.size(2)]
                        attn = attn + mask_to_use
                    else:
                        pass
                attn = F.softmax(attn, dim=-1)
                attn_output = torch.matmul(attn, v)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2))
            
            if mask is not None:
                if mask.dim() == 2:
                    mask_len = min(mask.size(0), ctx)
                    mask_to_apply = mask[:mask_len, :mask_len]
                    attn[:, :, :mask_len, :mask_len] = attn[:, :, :mask_len, :mask_len] + mask_to_apply
                elif mask.dim() == 3:
                    mask_len = min(mask.size(1), ctx)
                    mask_to_apply = mask[:, :mask_len, :mask_len]
                    attn[:, :, :mask_len, :mask_len] = attn[:, :, :mask_len, :mask_len] + mask_to_apply.unsqueeze(1)
                elif mask.dim() == 4:
                    mask_q_len = min(mask.size(2), ctx)
                    mask_k_len = min(mask.size(3), k.size(2))
                    attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len] + mask[:, :, :mask_q_len, :mask_k_len]
                
            attn = F.softmax(attn, dim=-1)
            attn_output = torch.matmul(attn, v)
        
        output = attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims)
        return self.out(output)


class AdaptiveUpdateAttention(nn.Module):
    """Attention implementation with content-dependent update frequencies."""
    def __init__(self, dims: int, head: int, max_dist=256):
        super().__init__()
        self.query_module = QueryModule(dims, head)
        self.key_module = KeyModule(dims, head)
        self.value_module = ValueModule(dims, head)
        self.combiner = AttentionCombiner(dims, head)
        self.max_dist = max_dist
        self.head = head
        self.dims = dims

        self.key_update_predictor = nn.Sequential(
            Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid()
        )

        self.value_update_predictor = nn.Sequential(
            Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid()
        )

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
        """Modified forward method to match FocusA's interface"""
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

class AdaptiveSpan(nn.Module):
    def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
        super().__init__()
        self.head = head
        self.max_dist = max_dist
        self.dims = dims
        self.temp_scale = temp_scale
        self.sharpen = sharpen
        self.span_scale = nn.Parameter(torch.tensor(1.0))

        self.head_dim = dims // head
        self.register_buffer("scale", torch.tensor(self.head_dim**-0.25))

    @autocast('cuda', enabled = True)
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

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            temperature = (
                1.0 + self.temp_scale * (1.0 - span_mean)
                if self.sharpen
                else 0.5 + self.temp_scale * span_mean
            )
            scores = torch.matmul(q, k.transpose(-2, -1))
            weights = torch.softmax((scores / temperature) * self.scale, dim=-1)
            out = torch.matmul(weights, v)
            out = out.permute(0, 2, 1, 3).reshape(batch, eff_span, self.dims)

        return out, weights

class IntegratedAttention(nn.Module):
    def __init__(self, ctx, dims, head, max_dist=512, win_size=256, max_span=384, temp_scale=0.01,):
        super().__init__()
        self.head = head
        self.max_dist = max_dist
        self.dims = dims
        self.max_span = max_span
        self.sliding_window = win_size
        self.temp_scale = 0.01
        self.sharpen = True
        self.head_dim = dims // head
        self.batch = None

        self.refiner = Refiner(
            states=10000, actions=10, alpha=0.1, gamma=0.9, epsilon=0.1
        )
        self.span_pred = Predictor(dims=dims)
        self.attn_local = AdaptiveSpan(
            dims=dims, head=head, max_dist=max_dist, sharpen=True, temp_scale=0.01
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
        if mask is None:
            mask = self.mask
            
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
        with torch.no_grad():
            safe_output = output.clamp(min=1e-10)
            entropy = -(safe_output * torch.log(safe_output)).sum(-1).mean()
            coverage = (output > 0.01).float().mean()
            return float(coverage - 0.1 * entropy)

    def extract(self, x):
        with torch.no_grad():
            mean_state = x.mean(dim=(0, 1))
            var_state = x.var(dim=(0, 1), unbiased=False)
            state = torch.cat([mean_state, var_state])
            state_id = self.discretize(state.cpu().numpy())
        return state_id

    def discretize(self, state):
        bins = np.linspace(-1, 1, num=10)
        state_discrete = np.digitize(state, bins)
        state_hash = hash(tuple(state_discrete))
        state_id = state_hash % (self.refiner.states - 1)
        return state_id

    def action_scale(self, action):
        span_value = action / (self.refiner.actions - 1)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        span_scale = torch.tensor([span_value], device=device, dtype=dtype)
        return span_scale
    
    @autocast('cuda', enabled = True)
    def _focus(self, query, key, value, span_scale, mask):
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
            d_k = dims // self.head
            scale_factor = 1 / math.sqrt(d_k)

            q = q_span.view(batch, ctx, self.head, -1).transpose(1, 2)
            k = k_span.view(batch, ctx, self.head, -1).transpose(1, 2)
            v = v_span.view(batch, ctx, self.head, -1).transpose(1, 2)

            if self.sharpen:
                temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
            else:
                temperature = 0.5 + self.temp_scale * span_scale.mean().item()
            attn_scores = (
                torch.matmul(q, k.transpose(-2, -1)) * scale_factor / temperature
            )
            if mask.size(-2) != attn_scores.size(-2) or mask.size(
                -1
            ) != attn_scores.size(-1):

                mask_q_len = min(mask.size(-2), attn_scores.size(-2))
                mask_k_len = min(mask.size(-1), attn_scores.size(-1))
                resized_mask = torch.ones(
                    (
                        batch,
                        self.head,
                        attn_scores.size(-2),
                        attn_scores.size(-1),
                    ),
                    device=mask.device,
                    dtype=mask.dtype,
                )
                resized_mask[:, :, :mask_q_len, :mask_k_len] = mask[
                    :, :, :mask_q_len, :mask_k_len
                ]
                mask = resized_mask

            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = (
                attn_out.transpose(1, 2).contiguous().view(batch, ctx, -1)
            )

            diff = torch.abs(attn_out - prev_attn).mean()
            dynamic_threshold = threshold + s_factor * diff

            if diff < dynamic_threshold:
                break

            prev_attn = attn_out
            query = query + attn_out
            iteration += 1
        return attn_out, attn_weights
    
    @autocast('cuda', enabled = True)
    def slide_win(self, x, win_size, span_len, span_scale, mask):
        batch, ctx, dims = x.size()
        self.batch = batch
        num_windows = (ctx + win_size - 1) // win_size
        output = torch.zeros_like(x)
        device = x.device
        default_mask = None

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

            if mask is not None:
                if mask.dim() == 4:
                    window_mask = mask[:, :, start_idx:end_idx, key_start:key_end]
                    if window_mask.size(1) == 1:
                        window_mask = window_mask.expand(-1, self.head, -1, -1)
                else:
                    if (
                        default_mask is None
                        or default_mask.size(-2) != window_size
                        or default_mask.size(-1) != span_size
                    ):
                        default_mask = torch.ones(
                            (batch, self.head, window_size, span_size),
                            device=device,
                            dtype=torch.bool,
                        )
                    window_mask = default_mask
            else:
                if (
                    default_mask is None
                    or default_mask.size(-2) != window_size
                    or default_mask.size(-1) != span_size
                ):
                    default_mask = torch.ones(
                        (batch, self.head, window_size, span_size),
                        device=device,
                        dtype=torch.bool,
                    )
                window_mask = default_mask

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
    def __init__(self, dims: int, head: int, dropout: float, act: str, debug=False, cross_attention=False):
        if debug is True:
            print(f"Residual check:{dims} {head} {dropout} {act}")

        super().__init__()

        self.dims = dims
        self.head = head
        self.dropout = dropout
        self.act = act
        self.cross_attention = cross_attention

        act_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }
        act = act_map.get(act, nn.GELU())
        
        self.attna = MultiheadA(dims=dims, head=head)
        #self.attnb = IntegratedAttention(dims=dims, head=head) if IntegratedAttention else None
        self.attnc = MultiheadA(dims=dims, head=head) if cross_attention else None
    
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            Linear(in_features=dims, out_features=dims * 4, bias=True),
            act,
            nn.Dropout(p=dropout),
            Linear(in_features=dims * 4, out_features=dims, bias=True))

        self.lna = LayerNorm(normalized_shape=dims)
        self.lnb = LayerNorm(normalized_shape=dims) 
        self.lnc = LayerNorm(normalized_shape=dims) if cross_attention else None

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attna(self.lna(x), mask=mask, kv_cache=kv_cache)[0]
        if self.attnc is not None:
            x = x + self.attnc(self.lnc(x), xa, kv_cache=kv_cache)[0] # type: ignore # head mask revisit
        x = x + self.mlp(self.lnb(x))
        return x
    
class AudioEncoder(nn.Module):
    def __init__( self, mels: int, ctx: int, dims: int, head: int, layerA: int, layerB: int, checkpoint: bool, dropout: float, act: str,  
                 scale_embedding, debug=None):
        if debug == 1:
            print(
                f"AudioEncoder check: {mels} {ctx} {dims} {head} {checkpoint} {dropout} {act} {layerA} {layerB}"
            )
        super().__init__()
        self.ctx = ctx
        self.dims = dims
        self.head = head
        self.layerA = layerA
        self.layerB = layerB
        self.checkpoint = checkpoint
        self.dropout = dropout
        self.act = act
        self.scale_embedding = scale_embedding
        self.rotary = rotary(dims=dims, head=head)

        act_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }
        act = act_map.get(act, nn.GELU())

        self.conv1 = Conv1d(mels, dims, kernel_size=3, padding=1)
        self.conv2 = Conv1d(dims, dims, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(ctx, dims))

        self.blockA = ( nn.ModuleList( modules=[ Residual(dims=dims, head=head, dropout=dropout, act=act, debug=debug) 
                                                for _ in range(layerA) ] ) if layerA > 0 else None )

        self.blockB = ( nn.ModuleList( modules=[ IntegratedAttention(ctx=ctx, dims=dims, head=head)
                                                for _ in range(layerB) ] ) if layerB > 0 else None )
        self.ln_enc = LayerNorm(dims)

    def forward(self, x: Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = x + self.rotary(x)
        x = x * self.scale_embedding

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        x = ( checkpoint(self._forward, x, use_reentrant=True) if self.checkpoint else x ) 

        for block in chain(self.blockA or [], self.blockB or []):
            if self.checkpoint:
                x = checkpoint(block, x, use_reentrant=True)
            else:
                result = block(x)
                if isinstance(result, tuple):
                    x = result[0]
                else:
                    x = result
        x = self.ln_enc(x)
        return x
    
    def _forward(self, x: Tensor):
        return x

class TextDecoder(nn.Module):
    def __init__( self, vocab: int, ctx: int, dims: int, head: int, layerA: int, layerB: int, checkpoint: bool, dropout: float, act: str,  debug=None):
        if debug == 2: print( f"TextDecoder check: {vocab} {ctx} {dims} {head} {checkpoint} {dropout} {act} {layerA} {layerB}" )  # noqa: E701
        super().__init__() 
        self.checkpoint = checkpoint 
        self.dims = dims 
        self.head = head 
        self.layerA = layerA 
        self.layerB = layerB 
        self.dropout = dropout 
        self.act = act 
        
        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims) 
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02) 
        
        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims)) 
        nn.init.normal_(tensor=self.positional_embedding, mean=0.0, std=0.02) 

        self.ln_dec = LayerNorm(normalized_shape=dims) 
        
        self.blockA = ( nn.ModuleList( modules=[ Residual(dims=dims, head=head, dropout=dropout, act=act ) 
                                                for _ in range(layerA) ] ) if layerA > 0 else None ) 
        
        self.blockB = ( nn.ModuleList( modules=[ IntegratedAttention(ctx=ctx, dims=dims, head=head)
                                                for _ in range(layerB) ] ) if layerB > 0 else None )
        
        mask = torch.empty(ctx, ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in chain(self.blockA or [], self.blockB or []):
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln_dec(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        return logits

class Echo(nn.Module):
    def __init__(self, param: Dimensions, debug=None):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.param = param
        self.to(device)

        self.encoder = AudioEncoder(
            self.param.mels,
            self.param.audio_ctx,
            self.param.audio_dims,
            self.param.audio_head,
            self.param.audio_layerA,
            self.param.audio_layerB,
            self.param.audio_checkpoint,
            self.param.audio_dropout,
            self.param.audio_act,
            self.param.scale_embedding,

        )
        self.decoder = TextDecoder(
            self.param.vocab,
            self.param.text_ctx,
            self.param.text_dims,
            self.param.text_head,
            self.param.text_layerA,
            self.param.text_layerB,
            self.param.text_checkpoint,
            self.param.text_dropout,
            self.param.text_act,
        )

        all_head = torch.zeros(
            self.param.text_layerA, self.param.text_head, dtype=torch.bool
        )
        all_head[self.param.text_layerA // 2 :] = True
        self.register_buffer("alignment_head", all_head.to_sparse(), persistent=False)

    def _init_weights(self, module):
        std = 0.02
        self.init_counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "Embedding": 0}

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            if isinstance(module, Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            if isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                self.init_counts["LayerNorm"] += 1
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
                self.init_counts["Embedding"] += 1
                
    def init_weights(self):  # noqa: F811
        print("Initializing all weights")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            print(f"{module_type}: {count}")

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

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss = None
        if self.training:
            audio_features = self.encoder(mel)
            logits = self.decoder(tokens, audio_features)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1))
            self.update_freq(loss=loss.item())
        else:
            audio_features = self.encoder(mel)
            logits = self.decoder(tokens, audio_features)
        return {"loss": loss, "logits": logits}

    
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

        # Apply hooks to both encoder and decoder if needed
        if self.blockB and any(isinstance(m, AdaptiveUpdateAttention) for m in self.blockB):
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
            if isinstance(module, (rotary)):
                module.update_freq(new_freq=self.new_freq)

    def generate(self, mel: torch.Tensor, max_length: int = 512) -> torch.Tensor:
        audio_features = self.encoder(mel)
        return self.decoder.generate(audio_features, max_length=max_length)

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
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(
            waveform
        )
    return waveform.flatten()

def pad(array, target_length, axis=-1, dtype: torch.dtype = torch.float32):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(ndarray=array).to(dtype=dtype)
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
        norm='slaney',
        normalized=True,
        power=2.0,
        center=True, 
        window_fn=torch.hann_window,
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
    def __init__(self, tokenizer, audio_ctx, text_ctx, mels, n_fft=1024, hop_length=160, sample_rate=16000, device="cpu"):
        self.tokenizer = tokenizer
        self.text_ctx = text_ctx
        self.audio_ctx = audio_ctx
        self.sample_rate = sample_rate
        self.mels = mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.decoder_start_token_id = tokenizer.bos_token_id 
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        batch = len(features)

        max_time_frames = (
            ctx_to_samples(audio_ctx=self.audio_ctx, hop_length=self.hop_length)
            // self.hop_length
        )
        batch_audio = torch.zeros(
            size=(batch, self.mels, max_time_frames),
            dtype=torch.float32,
            device=self.device,
        )
        batch_input_ids = torch.full(
            size=(batch, self.text_ctx),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        batch_labels = torch.full(
            size=(batch, self.text_ctx),
            fill_value=-100,
            dtype=torch.long,
            device=self.device,
        )

        for i, feature in enumerate(iterable=features):

            audio = process_audio(
                audio=feature["audio"],
                audio_ctx=self.audio_ctx,
                mels=self.mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                sr=self.sample_rate,
            )
            time_frames = audio.shape[-1]
            transcript = feature["transcription"]
            encoded_input = self.tokenizer.encode(transcript)
            encoded_label = self.tokenizer.encode(transcript)
            decoder_input = [self.decoder_start_token_id] + encoded_input
            labels = encoded_label + [self.tokenizer.eos_token_id]
            decoder_input = decoder_input[: self.text_ctx] + [self.pad_token_id] * (
                self.text_ctx - len(decoder_input)
            )
            labels = labels[: self.text_ctx] + [-100] * (self.text_ctx - len(labels))
            batch_input_ids[i, : len(decoder_input)] = torch.tensor(data=decoder_input, dtype=torch.long)
            batch_labels[i, : len(labels)] = torch.tensor(data=labels, dtype=torch.long)
            batch_audio[i, :, :time_frames] = torch.tensor(data=audio, dtype=torch.float32)

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
    wer = 100 * metric.compute(predictions=pred_str, references=label_str) # type: ignore
    return {"wer": wer}


# %%


def train_and_evaluate(model, tokenizer, train_loader, eval_loader, optimizer, scheduler, loss_fn,
                        max_steps=10000, device='cuda', accumulation_steps=1, clear_cache=True,
                        log_interval=10, eval_interval=100, save_interval=1000,
                        checkpoint_dir="checkpoint_dir", log_dir="log_dir"):
    model.to(device)
    global_step = 0
    scaler = torch.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    train_iterator = iter(train_loader)
    total_loss = 0
    step_in_report = 0
    dataset_epochs = 0

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
            # model.adjust_freq(loss=loss.item())
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
            writer.add_scalar(tag='Loss/train', scalar_value=total_loss / (global_step + 1), global_step=global_step)
            lr = scheduler.get_last_lr()[0]
            writer.add_scalar(tag='LearningRate', scalar_value=lr, global_step=global_step)
            writer.add_scalar(tag='SamplesPerSec', scalar_value=samples_per_sec, global_step=global_step)

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
                # for eval_batch in tqdm(eval_loader, desc=f"Evaluating (Step {global_step})", leave=True, colour='green'):
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

            eval_time = time.time() - eval_start_time
            loss_avg = eval_loss / batch_count if batch_count > 0 else 0
            predictions = {"predictions": np.array(all_predictions, dtype=object), "label_ids": np.array(all_labels, dtype=object)}
            metrics = compute_metrics(pred=predictions, tokenizer=tokenizer)

            writer.add_scalar('Loss/eval', loss_avg, global_step)
            writer.add_scalar('WER', metrics['wer'], global_step)
            writer.add_scalar('EvalSamples', total_samples, global_step)
            writer.add_scalar('EvalTimeSeconds', eval_time, global_step)

            lr = scheduler.get_last_lr()[0]
            print(f" • WER:{metrics['wer']:.2f}% • Loss:{loss_avg:.4f} • LR:{lr:.8f}")

            logging.info(f"EVALUATION STEP {global_step} - WER: {metrics['wer']:.2f}%, Loss: {loss_avg:.4f}, LR: {lr:.8f}")
            #scheduler.step()
            model.train()

        if global_step % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            #print(f"Model saved at step {global_step} to {checkpoint_path}")
            logging.info(f"Model saved at step {global_step} to {checkpoint_path}")
            
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        global_step += 1
        step_in_report += 1
        
        avg_loss = total_loss / (global_step + 1)
        postfix_dict = {
            'loss': f'{avg_loss:.4f}',
            'lr': f'{lr:.6f}',
            'WER': f'{metrics["wer"]:.4f}' if 'wer' in metrics else 'N/A',
            'samp/sec': f'{samples_per_sec:.1f}'
        }
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
    log_dir = os.path.join("./output/logs", datetime.now().strftime(format="%m-%d_%H"))
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
   
    debug = None
    param = Dimensions(
        mels=128,
        audio_ctx=1500,
        audio_head=4,
        audio_layerA=4,
        audio_layerB=2,
        audio_dims=512,
        audio_act="gelu",
        audio_checkpoint=False,
        audio_dropout=0.001,
        scale_embedding=1.0,
        audio_debug = debug,
        vocab=51865,
        text_ctx=448,
        text_head=4,
        text_layerA=4,
        text_layerB=0,
        text_dims=512,
        text_act="gelu",
        text_checkpoint=False,
        text_dropout=0.001,
        text_debug = debug,
         )
    
    model = Echo(param=param).to('cuda')
    model.init_weights()

    DataCollator = DataCollator(tokenizer=tokenizer, 
                                audio_ctx=param.audio_ctx, 
                                text_ctx=param.text_ctx, 
                                mels=param.mels, 
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
    


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, total_iters=10000, last_epoch=-1)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
    train_and_evaluate(model=model, 
        tokenizer=tokenizer, 
        train_loader=train_dataloader, 
        eval_loader=eval_dataloader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        loss_fn=loss_fn, 
        max_steps=10000,
        device='cuda', 
        accumulation_steps=1, 
        clear_cache=False, 
        log_interval=10, 
        eval_interval=500, 
        save_interval=10000, 
        checkpoint_dir=checkpoint_dir, 
        log_dir=log_dir
        )


# %%
from tensorboard import program
log_dir = "./output/logs" 
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()
print(f"TensorBoard started at {url}")





```


