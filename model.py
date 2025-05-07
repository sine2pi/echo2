import os
import warnings
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional, Dict, Union, List, Tuple
from functools import partial
import gzip
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from datetime import datetime
from datasets import load_dataset, Audio
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperFeatureExtractor, WhisperTokenizer
import evaluate
import transformers
from dataclasses import dataclass
from itertools import chain

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
transformers.utils.logging.set_verbosity_error()
device = torch.device(device="cuda:0")
dtype = torch.float32

torch.set_default_dtype(dtype)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}


extractor = None
tokenizer = None
optimizer = None 
scheduler = None

def set_extractor_and_tokenizer(extractor_, tokenizer_):
    global extractor, tokenizer
    extractor = extractor_
    tokenizer = tokenizer_

@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    decoder_idx: int
    mels: int
    audio_ctx: int
    audio_dims: int
    audio_head: int
    encoder_idx: int
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
    act: str 


class MaxFactor(torch.optim.Optimizer):
    """
    MaxFactor optimizer: A memory-efficient optimizer with max-based normalization
    for training speech recognition models.
    
    Key features:
    - Factorized second moments (memory efficient)
    - Max-based normalization (better for attention & rotary params)
    - Infinity norm clipping (prevents extreme updates)
    - Per-parameter adaptive learning rates
    
    Args:
        params: Model parameters or param groups
        lr: Learning rate (default: 0.01)
        beta2_decay: Power for step-size decay (default: -0.8)
        eps: Small constants for numerical stability (default: (1e-10, 1e-4))
        d: Update scale control factor (default: 1.0)
        weight_decay: Weight decay factor (default: 0.01)
        gamma: EMA decay rate for non-factorized tensors (default: 0.99)
        max_norm: Whether to use max normalization (default: True)
        min_lr: Minimum learning rate (default: 1e-7)
    """
    def __init__(self, params, lr=0.01, beta2_decay=-0.8, eps=(1e-10, 1e-4), 
                 d=1.0, weight_decay=0.01, gamma=0.99, max_norm=True, min_lr=1e-7, scalar_boost=2.0):
        
        defaults = dict(
            lr=lr, 
            beta2_decay=beta2_decay, 
            eps=eps, 
            d=d, 
            weight_decay=weight_decay, 
            gamma=gamma, 
            max_norm=max_norm, 
            min_lr=min_lr,
            scalar_boost=scalar_boost
        )
        super().__init__(params=params, defaults=defaults)
        print(f"MaxFactor optimizer initialized with lr={lr}, beta2_decay={beta2_decay}")

    def _get_lr(self):
        """Return the current learning rates as a dictionary."""
        return {i: group['lr'] for i, group in enumerate(self.param_groups)}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps1, eps2 = group["eps"]
            min_lr = group.get("min_lr", 1e-7)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.float()
                state = self.state[p]
                
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, device=p.device)
                    
                    if p.dim() > 1:
                        row_shape, col_shape = list(p.shape), list(p.shape)
                        row_shape[-1], col_shape[-2] = 1, 1
                        state["row_var"] = torch.zeros(row_shape, device=p.device)
                        state["col_var"] = torch.zeros(col_shape, device=p.device)
                    
                    state["v"] = torch.zeros_like(p)
                
                state["step"] += 1
                step_float = state["step"].item()
                
                one_minus_beta2_t = min(0.999, max(0.001, step_float ** group["beta2_decay"]))
                rho_t = max(min_lr, min(group["lr"], 1.0 / (step_float ** 0.5)))
                param_scale = (p.norm() / (p.numel() ** 0.5 + 1e-12)).item()
                alpha = max(eps2, param_scale) * rho_t
                
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                
                if p.dim() > 1:
                    row_var = state["row_var"]
                    col_var = state["col_var"]
                    
                    row_mean = torch.norm(grad, dim=-1, keepdim=True).square_()
                    row_mean.div_(grad.size(-1) + eps1)
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    
                    col_mean = torch.norm(grad, dim=-2, keepdim=True).square_()
                    col_mean.div_(grad.size(-2) + eps1)
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    
                    var_estimate = row_var @ col_var
                    
                    if group["max_norm"]:
                        max_row_var = row_var.max(dim=-2, keepdim=True)[0]
                        var_estimate.div_(max_row_var.clamp_(min=eps1))
                
                else:
                    vi = state["v"]
                    vi.mul_(group["gamma"]).add_(grad.square_(), alpha=1 - group["gamma"])
                    var_estimate = vi


                if p.numel() == 1:
                    update = grad / (var_estimate.sqrt() + eps1)
                    scalar_boost = group.get("scalar_boost", 2.0)
                    p.add_(-alpha * scalar_boost * update)
                else:
                    update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_().mul_(grad)
                    
                    inf_norm = torch.norm(update, float('inf'))
                    if inf_norm > 0:
                        update.div_(inf_norm.clamp_(min=eps1))
                    
                    denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * group["d"]))
                    
                    if p.dim() > 1 and group["max_norm"]:
                        max_vals = update.abs().max(dim=-1, keepdim=True)[0]
                        p.add_(-alpha / denom * update.sign() * max_vals)
                    else:
                        p.add_(-alpha / denom * update)
                    
            return loss
    
def create_specialized_maxfactor(model, lr=0.01, weight_decay=0.01, max_norm=True):
    base_lr = 0.01
    regular_params = []
    attention_factors = []
    rotary_params = []
    positional_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'factor' in name and ('attna' in name or 'attnb' in name):
            attention_factors.append(param)
        elif 'rotary' in name and 'inv_freq' in name:
            rotary_params.append(param)
        elif 'positional_embedding' in name:
            positional_params.append(param)
        else:
            regular_params.append(param)
    
    param_groups = [
        {'params': regular_params, 'lr': base_lr, 'weight_decay': 0.01, 'max_norm': True},
        {'params': attention_factors, 'lr': base_lr * 10.0, 'weight_decay': 0.0, 'max_norm': True},
        {'params': rotary_params, 'lr': base_lr * 50.0, 'weight_decay': 0.0, 'max_norm': True},
        {'params': positional_params, 'lr': base_lr * 5.0, 'weight_decay': 0.005, 'max_norm': True}
    ]
    
    return MaxFactor(param_groups, lr=base_lr, beta2_decay=-0.8, eps=(1e-10, 1e-4), d=1.0, weight_decay=0.01, gamma=0.99, max_norm=True, min_lr=1e-7)   

def get_tracked_parameters(model, param_paths=None):
    if param_paths is None:
        param_paths = {
            "blend_sw": "encoder.blend_sw",
        }
    result = {}
    for name, path in param_paths.items():
        parts = path.split('.')
        param = model
        for part in parts:
            param = getattr(param, part)
        
        try:
            if isinstance(param, torch.Tensor):
                if param.numel() == 1:
                    result[name] = param if not param.requires_grad else param
                else:
                    result[name] = param.sum()
            else:
                result[name] = float(param) if hasattr(param, "__float__") else str(param)
        except Exception as e:
            result[name] = f"Error: {str(e)}"
    
    return result

def plot_waveform_and_spectrogram(x=None, w=None, sample_idx=0, sr=16000, title="Waveform and Spectrogram"):
    """Plot waveform and/or spectrogram based on available inputs."""
    if x is not None and w is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
    
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr

        fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
        axs[0].plot(t, w_np, color="tab:blue")
        axs[0].set_title("Waveform")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude")

        axs[1].imshow(x_np.T, aspect="auto", origin="lower", cmap="magma")
        axs[1].set_title("Spectrogram")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Mel Bin")
        plt.tight_layout()
        plt.show()
    elif x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.imshow(x_np.T, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("Spectrogram")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel Bin")
        plt.tight_layout()
        plt.show()     
    elif w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.plot(t, w_np, color="tab:blue")
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("No data to plot. Please provide at least one input tensor.")

def shift_with_zeros(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    input_ids = labels.new_zeros(labels.shape)
    input_ids[:, 1:] = labels[:, :-1].clone()   
    return input_ids

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)
    
class RMSNorm(nn.RMSNorm):
    def __init__(self, dims: Union[int, Tensor, List, Tuple], eps = 1e-8, elementwise_affine = True, device=torch.device(device="cuda:0"), dtype=torch.float32):
        tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}

        if isinstance(dims, int):
            self.normalized_shape = (dims,)  
        else:
            self.normalized_shape = tuple(dims) 
        super().__init__(normalized_shape=dims, eps=eps, elementwise_affine=elementwise_affine)
        
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **tox))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
            
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, 
            self.weight.to(x.device, x.dtype),
            None if self.bias is None else self.bias.to(x.device, x.dtype)
        )
    
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)      


class Rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, learned_freq=True):
        super().__init__()
        self.dims = dims
        self.inv_freq = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
            requires_grad=learned_freq
        )
        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))  

    def forward(self, positions):
        if isinstance(positions, int):
            t = torch.arange(positions, device=self.inv_freq.device).float()
        else:
            t = positions.float().to(self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs + self.bias[:freqs.shape[0]]
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
        x1 = torch.view_as_complex(x1)
        x1 = x1 * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        return torch.cat([x1.type_as(x), x2], dim=-1)
    
    def precompute_freqs_cis(dims: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dims, 2)[: (dims // 2)].float() / dims)) 
        t = torch.arange(end, device=freqs.device)  
        freqs = torch.outer(t, freqs).float()  
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
        return freqs_cis
    
    def reset_parameters(self):
        nn.init.orthogonal_(tensor=self.matrix)
        nn.init.zeros_(tensor=self.thetas)

    def q_rotation(self, x, theta, u, v):
        x = x.to(self.device, self.dtype)
        theta = theta.to(self.device, self.dtype) if not isinstance(theta, (int, float)) else theta
        u = u.to(self.device)
        v = v.to(self.device)
        u = u / torch.norm(u)
        v = v / torch.norm(v)
        half_theta = theta / 2
        cos_ht = torch.cos(half_theta)
        sin_ht = torch.sin(half_theta)
        q = torch.cat([cos_ht.unsqueeze(0), sin_ht * u])
        q_ = torch.cat([cos_ht.unsqueeze(0), -sin_ht * u])
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

    def rotate(self, x):
        rotate = int(torch.round(self.head_dim // 2))
        for k in range(rotate):
            i, j = self.pairs[k].long()
            theta = self.thetas[k]
            G = self.rotation_matrix(dims=self.head_dim, i=i.item(), j=j.item(), theta=theta)
            x = x @ G
        return x


class MultiheadA(nn.Module):
    def __init__(self, dims: int, head: int, rotary):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.out = Linear(dims, dims)

        self._rotary = rotary
        self.factor = nn.Parameter(torch.tensor(0.001))
        
    @property
    def rotary(self):
        return self._rotary

    def forward(self, x: Tensor, xa: Optional[Tensor] = None,  mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        batch, ctx, dims = x.shape

        q, k, v = self.q(x if xa is None else xa), self.k(x if xa is None else xa), self.v(x if xa is None else xa)
        wv, qk = self._attention(q, k, v, mask)
        return self.out(wv), qk
    
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v = None, mask = None):
        factor = self.factor
        batch, ctx, dims = q.shape
        self.freq = self._rotary(ctx)
        scale = (dims // self.head) ** -0.25
        
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        q = self._rotary.apply_rotary(q, self.freq)
        k = self._rotary.apply_rotary(k, self.freq)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        mask = torch.triu(torch.ones(ctx, ctx), diagonal=1)
        scaled_mask = torch.where(mask == 1, torch.tensor(0.0), torch.tensor(1.0)).to(q.device, q.dtype)

        token_ids = k[:, :, :, 0]
        scaled_zero = torch.ones_like(token_ids)
        zero_factor = torch.clamp(F.softplus(factor), min=0.00001, max=0.1)
        scaled_zero[token_ids.float() == 0] = zero_factor.to(q.device, q.dtype)
        scaling_factors = scaled_mask.unsqueeze(0) * scaled_zero.unsqueeze(-2).expand(qk.shape)
        qk *= scaling_factors
        qk = qk.float()
        w = F.softmax(qk, dim=-1)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return out, qk.detach()


class MultiheadB(nn.Module):
    def __init__(self, dims: int, head: int, rotary):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.out = Linear(dims, dims)

        self._rotary = rotary
        self.factor = nn.Parameter(torch.tensor(0.001))
        
    @property
    def rotary(self):
        return self._rotary

    def forward(self, x: Tensor, xa: Optional[Tensor] = None,  mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        batch, ctx, dims = x.shape

        q, k, v = self.q(x if xa is None else xa), self.k(x if xa is None else xa), self.v(x if xa is None else xa)
        wv, qk = self._attention(q, k, v, mask)
        return self.out(wv), qk
    
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v = None, mask = None):
        factor = self.factor
        batch, ctx, dims = q.shape
        self.freq = self._rotary(ctx)
        scale = (dims // self.head) ** -0.25
        
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)

        q = self._rotary.apply_rotary(q, self.freq)
        k = self._rotary.apply_rotary(k, self.freq)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)

        mask = torch.triu(torch.ones(ctx, ctx), diagonal=1)
        scaled_mask = torch.where(mask == 1, torch.tensor(0.0), torch.tensor(1.0)).to(q.device, q.dtype)

        token_ids = k[:, :, :, 0]
        scaled_zero = torch.ones_like(token_ids)
        zero_factor = torch.clamp(F.softplus(factor), min=0.00001, max=0.1)
        scaled_zero[token_ids.float() == 0] = zero_factor.to(q.device, q.dtype)
        scaling_factors = scaled_mask.unsqueeze(0) * scaled_zero.unsqueeze(-2).expand(qk.shape)
        qk *= scaling_factors
        qk = qk.float()
        w = F.softmax(qk, dim=-1)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return out, qk.detach()

class Residual(nn.Module):
    def __init__(self, dims: int, head: int, cross_attention = False, act = "relu", rotary=None):
        super().__init__()
        
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.cross_attention = cross_attention
        self.dropout = 0.1
        self._rotary = rotary

        self.blend_xa = nn.Parameter(torch.tensor(0.5)) 

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.attna = MultiheadA(dims=dims, head=head, rotary=self._rotary)
        self.attnb = MultiheadA(dims=dims, head=head, rotary=self._rotary) if cross_attention else None
    
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), self.act, Linear(mlp, dims))
        self.lna = RMSNorm(dims=dims)    
        self.lnb = RMSNorm(dims=dims) if cross_attention else None
        self.lnc = RMSNorm(dims=dims) 

    def forward(self, x, xa=None, mask=None, kv_cache=None):

        r = x
        x = x + self.attna(self.lna(x), mask=mask, kv_cache=kv_cache)[0]
        if self.attnb and xa is not None:
            cross_out = self.attnb(self.lnb(x), xa, kv_cache=kv_cache)[0]
            blend = torch.sigmoid(self.blend_xa)
            x = blend * x + (1 - blend) * cross_out
        x = x + self.mlp(self.lnc(x))
        x = x + r
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
     


class AudioEncoder(nn.Module):
    def __init__(self, mels: Optional[Tensor], ctx: Optional[Tensor], dims: Optional[Tensor], head, layer, act, cross_attention = False, rotary=None, **tox):
        super().__init__()

        self.dropout = 0.1
        self._counter = 0
        self._rotary = rotary

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        self.act = act_map.get(act, nn.GELU())

        self.blend_sw = nn.Parameter(torch.tensor(0.5))

        self.ln_enc = RMSNorm(dims, **tox)
        self.register_buffer("positional_embedding", sinusoids(ctx, dims))

        self.se = nn.Sequential(
            Conv1d(mels, dims, kernel_size=3, padding=1), self.act,
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=2, dilation=2),
            Conv1d(dims, dims, kernel_size=3, stride=1, padding=1, groups=dims),
            Conv1d(dims, dims, kernel_size=1), SEBlock(dims, reduction=16), self.act,
            nn.Dropout(p=self.dropout), Conv1d(dims, dims, kernel_size=3, stride=1, padding=1)
        )
        self.we = nn.Sequential(
            nn.Conv1d(1, dims, kernel_size=11, stride=5, padding=5),
            nn.GELU(),
            nn.Conv1d(dims, dims, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(ctx),
        )

        self.blockA = (nn.ModuleList([
            Residual(dims=dims, head=head, cross_attention=cross_attention, rotary=self._rotary) 
            for _ in range(layer)]) if layer > 0 else None)
        
    def forward(self, x, w) -> Tensor:

        blend = torch.sigmoid(self.blend_sw)
        if self._counter < 1:
            plot_waveform_and_spectrogram(x=x, w=w)

        if x is not None:
            if w is not None:
                x = self.se(x).permute(0, 2, 1)
                w = self.we(w).permute(0, 2, 1)
                x = (x + self.positional_embedding).to(x.device, x.dtype)
                w = (w + self.positional_embedding).to(w.device, w.dtype) 
                x = blend * x + (1 - blend) * w
            else:
                x = self.se(x)
                x = x.permute(0, 2, 1)
                assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
                x = (x + self.positional_embedding).to(x.device, x.dtype)
                x = blend * x + (1 - blend) * x
        else:
            assert w is not None, "You have to provide either x or w"
            x = self.we(w).permute(0, 2, 1)
            assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            x = (x + self.positional_embedding).to(x.device, x.dtype)
            x = blend * x + (1 - blend) * x 
            
        for block in chain(self.blockA or []):
            x = block(x)

        self._counter += 1
        return self.ln_enc(x)
        


class TextDecoder(nn.Module):
    def __init__(self, vocab, ctx, dims, head, layer, cross_attention = False, rotary=None, factor=None, **tox):
        super().__init__()
        
        self.debug = False
        self.dropout = 0.1
        self._rotary = rotary
        self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

        self.positional_embedding = nn.Parameter(data=torch.empty(ctx, dims), requires_grad=True)
        self.ln_dec = RMSNorm(dims=dims)
        self.rotary = Rotary(dims=dims, max_ctx=ctx, learned_freq=True)
    
        self.blockA = (nn.ModuleList([
            Residual(dims=dims, head=head, cross_attention=cross_attention, rotary=self._rotary) for _ in range(layer)]) if layer > 0 else None)

        mask = torch.triu(torch.ones(ctx, ctx), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, xa, kv_cache=None) -> Tensor:

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (self.token_embedding(x) + self.positional_embedding[offset: offset + x.shape[-1]])
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        ctx = x.shape[1]
        freqs = self.rotary(ctx)
        x = self.rotary.apply_rotary(x, freqs)
        x = x.to(xa.dtype)

        for block in chain(self.blockA or []):
            x = block(x, xa=xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln_dec(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits


class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param  

        self.param_tracking_paths = {
            "att_rot": "shared.rotary_encoder.inv_freq",
            "dec_rot": "shared.rotary_decoder.inv_freq",
        }

        self.shared = nn.ModuleDict({
            "rotary_encoder": Rotary(dims=param.audio_dims // param.audio_head),
            "rotary_decoder": Rotary(dims=param.text_dims // param.text_head),
            })


        self.encoder = AudioEncoder(
            mels=param.mels,
            ctx=param.audio_ctx,
            dims=param.audio_dims,
            head=param.audio_head,
            layer=param.encoder_idx,
            act=param.act,
            rotary=self.shared["rotary_encoder"],
        )

        self.decoder = TextDecoder(
            vocab=param.vocab,
            ctx=param.text_ctx,
            dims=param.text_dims,
            head=param.text_head,
            layer=param.decoder_idx,
            rotary=self.shared["rotary_decoder"],
        )

        all_head = torch.zeros(self.param.decoder_idx, self.param.text_head, dtype=torch.bool)
        all_head[self.param.decoder_idx // 2 :] = True
        self.register_buffer("alignment_head", all_head.to_sparse(), persistent=False)

    def log_param_gradients(self):
        params_to_check = {
            "encoder.blend_sw": self.encoder.blend_sw,
            "rotary_encoder.inv_freq": self.shared["rotary_encoder"].inv_freq
        }
        
        for name, param in params_to_check.items():
            if param.grad is not None:
                print(f"{name} grad: {param.grad.norm().item():.6f}")
            else:
                print(f"{name} grad: None")

    def set_alignment_head(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        mask = torch.from_numpy(array).reshape(
            self.param.decoder_idx, self.param.text_head)
        self.register_buffer("alignment_head", mask.to_sparse(), persistent=False)

    def embed_audio(self, input_features: torch.Tensor):
        return self.encoder(input_features)

    def logits(self,input_ids: torch.Tensor, encoder_output: torch.Tensor):
        return self.decoder(input_ids, encoder_output)
    
    def forward(self, 
        decoder_input_ids=None,
        labels=None,
        input_features: torch.Tensor=None, 
        waveform: Optional[torch.Tensor]=None,
        input_ids=None, 
    ) -> Dict[str, torch.Tensor]:

        if labels is not None:
            if labels.shape[1] > self.param.text_ctx:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.param.text_ctx} tokens."
                )
            if input_ids is None:
                input_ids = shift_with_zeros(
                    labels, self.param.pad_token_id, self.param.decoder_start_token_id
                ).to('cuda')
            decoder_input_ids = input_ids
            if input_ids.shape[1] > self.param.text_ctx:
                raise ValueError(
                    f"Input IDs' sequence length {input_ids.shape[1]} cannot exceed the maximum allowed length of {self.param.text_ctx} tokens."
                )

        if input_features is not None:    
            if waveform is not None:
                encoder_output = self.encoder(x=input_features, w=waveform)
            else:
                encoder_output = self.encoder(x=input_features, w=None)
        elif waveform is not None:
            encoder_output = self.encoder(x=None, w=waveform)
        else:
            raise ValueError("You have to provide either input_features or waveform")
        logits = self.decoder(input_ids, encoder_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)
            
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output
        }

    @property
    def device(self):
        return next(self.parameters()).device

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []
        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.param.text_ctx:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

    def _init_weights(self, module):
        std = 0.02
        self.init_counts = {"Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "SEBlock": 0, "TextDecoder": 0, "AudioEncoder": 0, "Residual": 0,
                            "Multihead": 0, "MultiheadA": 0, "MultiheadB": 0, "MultiheadC": 0}

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            elif isinstance(module, Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                self.init_counts["LayerNorm"] += 1
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
                self.init_counts["RMSNorm"] += 1
            elif isinstance(module, MultiheadA):
                self.init_counts["MultiheadA"] += 1
            elif isinstance(module, MultiheadB):
                self.init_counts["MultiheadB"] += 1
            elif isinstance(module, Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, SEBlock):
                nn.init.ones_(module.fc[0].weight)
                nn.init.zeros_(module.fc[0].bias)
                nn.init.ones_(module.fc[2].weight)
                nn.init.zeros_(module.fc[2].bias)
                self.init_counts["SEBlock"] += 1
            elif isinstance(module, TextDecoder):
                self.init_counts["TextDecoder"] += 1
            elif isinstance(module, AudioEncoder):
                nn.init.xavier_uniform_(module.se[0].weight)
                nn.init.zeros_(module.se[0].bias)
                nn.init.xavier_uniform_(module.se[2].weight)
                nn.init.zeros_(module.se[2].bias)
                nn.init.xavier_uniform_(module.se[4].weight)
                nn.init.zeros_(module.se[4].bias)
                self.init_counts["AudioEncoder"] += 1
            elif isinstance(module, Residual):
                self.init_counts["Residual"] += 1    
                                                 
    def init_weights(self):
        print("Initializing all weights")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            print(f"{module_type}: {count}")

    


metric = evaluate.load(path="wer")

@dataclass
class DataCollator:

    def __call__(self, features: List[Dict[str, Union[List[int], Tensor]]]) -> Dict[str, Tensor]:
        global extractor, tokenizer
        decoder_start_token_id = tokenizer.bos_token_id
        pad_token_id = tokenizer.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError("The tokenizer does not have a bos_token_id. Please set it manually.")        
        batch = {}

        if "input_features" in features[0]:
            batch["input_features"] = torch.stack([f["input_features"] for f in features])
        if "waveform" in features[0]:
            batch["waveform"] = torch.stack([f["waveform"] for f in features])

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), 0)
        if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        batch["input_ids"] = shift_with_zeros(labels, pad_token_id, decoder_start_token_id)

        return batch
    
def prepare_dataset(batch, input_features=True, waveform=True):
    global extractor, tokenizer

    audio = batch["audio"]
    len = 1500 * 160
    wav = torch.tensor(audio["array"]).float()
    
    if wav.shape[-1] < len:
        wav = F.pad(wav, (0, len - wav.shape[-1]))
    else:
        wav = wav[..., :len]
    if waveform:
        batch["waveform"] = wav.unsqueeze(0)

    if input_features:
        features = extractor(wav.numpy(), sampling_rate=audio["sampling_rate"]).input_features[0]
        features = torch.tensor(features)
        pad_val = features.min().item() 
        features = torch.where(features == pad_val, torch.tensor(0.0, dtype=features.dtype), features)
        target_shape = (128, 1500)
        padded = torch.zeros(target_shape, dtype=features.dtype)
        padded[:, :features.shape[1]] = features[:, :target_shape[1]]
        batch["input_features"] = padded
    batch["labels"] = tokenizer(batch["transcription"], add_special_tokens=False).input_ids
    return batch

def compute_metrics(eval_pred, compute_result: bool = True):
    global extractor, tokenizer, model, optimizer, scheduler

    pred_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids

    if hasattr(pred_logits, "cpu"):
        pred_logits = pred_logits.cpu()
    if hasattr(label_ids, "cpu"):
        label_ids = label_ids.cpu()

    if isinstance(pred_logits, tuple):
        pred_ids = pred_logits[0]
    else:
        pred_ids = pred_logits

    if hasattr(pred_ids, "ndim") and pred_ids.ndim == 3:
        if not isinstance(pred_ids, torch.Tensor):
            pred_ids = torch.tensor(pred_ids)
        pred_ids = pred_ids.argmax(dim=-1)
        pred_ids = pred_ids.tolist()
    elif hasattr(pred_ids, "tolist"):
        pred_ids = pred_ids.tolist()

    if hasattr(label_ids, "tolist"):
        label_ids = label_ids.tolist()

    label_ids = [
        [tokenizer.pad_token_id if token == -100 else token for token in seq]
        for seq in label_ids
    ]

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    print("--------------------------------")
    print(f"Prediction: {pred_str[0]}")
    print(f"Label: {label_str[0]}")

    pred_flat = list(chain.from_iterable(pred_ids))
    labels_flat = list(chain.from_iterable(label_ids))
    mask = [i != tokenizer.pad_token_id for i in labels_flat]

    acc = accuracy_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m]
    )
    pre = precision_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    rec = recall_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    f1 = f1_score(
        [i for i, m in zip(labels_flat, mask) if m],
        [p for p, m in zip(pred_flat, mask) if m],
        average='weighted', zero_division=0
    )
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    tracked_params = get_tracked_parameters(model, model.param_tracking_paths)
    
    att_rot = tracked_params["att_rot"]
    dec_rot = tracked_params["dec_rot"]
  
    
    lr = optimizer.param_groups[0]['lr']
    metrics = {
        "wer": wer,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "rotaryA": att_rot,
        "rotaryB": dec_rot,
        "lr": lr,

    }
    return metrics

def create_model(param):
    model = Echo(param).to('cuda')
    model.init_weights()
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    return model

def setup_tokenizers(token):
    global extractor, tokenizer
    
    extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-small", 
        token=token, 
        feature_size=128,
        sampling_rate=16000, 
        do_normalize=True, 
        return_tensors="pt", 
        chunk_length=15, 
        padding_value=0.0
    )
    
    tokenizer = WhisperTokenizer.from_pretrained(
        "./tokenizer", 
        pad_token="0", 
        bos_token="0", 
        eos_token="0", 
        unk_token="0",
        token=token, 
        local_files_only=True, 
        pad_token_id=0
    )
    
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.decoder_start_token_id = 0

def prepare_datasets(token):
    dataset = load_dataset("google/fleurs", "en_us", token=token, trust_remote_code=True, streaming=False)
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000))
    
    def filter_func(x):
        return (0 < len(x["transcription"]) < 512 and
                len(x["audio"]["array"]) > 0 and
                len(x["audio"]["array"]) < 1500 * 160)
    
    dataset = dataset.filter(filter_func).shuffle(seed=4)
    print("Dataset size:", dataset["train"].num_rows, dataset["test"].num_rows)
    
    prepare_fn = partial(prepare_dataset, input_features=True, waveform=True)
    
    dataset = dataset.map(function=prepare_fn, remove_columns=list(next(iter(dataset.values())).features)
    ).with_format(type="torch")
    
    train_dataset = dataset["train"].shuffle(seed=4).flatten_indices()
    test_dataset = dataset["test"].shuffle(seed=4).take(200).flatten_indices()
    
    return train_dataset, test_dataset

def get_training_args(log_dir):
    return Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        tf32=True,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=10000,
        save_steps=10000,
        eval_steps=1000,
        warmup_steps=1000,
        num_train_epochs=1,
        logging_steps=100,
        logging_dir=log_dir,
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
    )



if __name__ == "__main__":
    
    param = Dimensions(
        mels=128,
        audio_ctx=1500,
        audio_head=4,
        encoder_idx=4,
        audio_dims=512,
        vocab=51865,
        text_ctx=512,
        text_head=4,
        decoder_idx=4,
        text_dims=512,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=0,
        act="gelu",
    )
    
    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H'))
    os.makedirs(name=log_dir, exist_ok=True)
    
    setup_tokenizers(token)
    model = create_model(param)
    train_dataset, test_dataset = prepare_datasets(token)
    training_args = get_training_args(log_dir)
   
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, weight_decay=0.0025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, last_epoch=-1)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        processing_class=extractor,
    )
        
    trainer.train()





