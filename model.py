
import pyworld as pw
import os
from torch.amp import autocast
import math, random
import warnings
import logging
import gzip
import base64
import torch
import torchaudio
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn, einsum, broadcast_tensors, Tensor
import numpy as np
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union, List, Tuple, Any
from functools import partial
from datetime import datetime
from datasets import load_dataset, Audio
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
import transformers
import evaluate
from dataclasses import dataclass
from math import pi, log

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
transformers.utils.logging.set_verbosity_error()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

extractor = None
tokenizer = None
optimizer = None
scheduler = None
model = None
Residual = None
MultiheadA = None

@dataclass
class Dimensions:
    vocab: int
    text_ctx: int
    text_dims: int
    text_head: int
    text_idx: int
    mels: int
    aud_ctx: int
    aud_dims: int
    aud_head: int
    aud_idx: int
    act: str
    debug: List[str]
    cross_attn: bool
    features: List[str]
    f0_rotary: bool

def plot_waveform(x=None, w=None, p=None, per=None, sample_idx=0, sr=16000, hop_length=160, 
                                 title="", markers=None, marker_labels=None, 
                                 show_voiced_regions=True, show_energy=False):
    num_plots = sum([x is not None, w is not None, p is not None, per is not None])
    if num_plots == 0:
        raise ValueError("No data to plot. Please provide at least one input tensor.")
    time_spans = []
    
    if w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        time_spans.append(len(w_np) / sr)
    if x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        time_spans.append(x_np.shape[0] * hop_length / sr)
    if p is not None:
        p_np = p[sample_idx].detach().cpu().numpy()
        if p_np.ndim > 1:
            p_np = p_np.squeeze()
        time_spans.append(len(p_np) * hop_length / sr)
    if per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        time_spans.append(len(per_np) * hop_length / sr)
    max_time = max(time_spans) if time_spans else 0
    fig, axs = plt.subplots(num_plots, 1, figsize=(14, 4*num_plots), sharex=True)
    if num_plots == 1:
        axs = [axs]
    if show_voiced_regions and per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        t_per = np.arange(len(per_np)) * hop_length / sr
        threshold = 0.5
        for ax in axs:
            for i in range(len(per_np)-1):
                if per_np[i] > threshold:
                    ax.axvspan(t_per[i], t_per[i+1], color='lightblue', alpha=0.2, zorder=0)
    current_ax = 0
    if w is not None:
        w_np = w[sample_idx].detach().cpu().numpy()
        if w_np.ndim > 1:
            w_np = w_np.squeeze()
        t = np.arange(len(w_np)) / sr
        axs[current_ax].plot(t, w_np, color="tab:blue")
        
        if show_energy:
            frame_length = hop_length
            hop_length_energy = hop_length // 2
            energy = []
            for i in range(0, len(w_np)-frame_length, hop_length_energy):
                frame = w_np[i:i+frame_length]
                energy.append(np.sqrt(np.mean(frame**2)))
            energy = np.array(energy)
            energy = energy / np.max(energy) * 0.8 * max(abs(w_np.min()), abs(w_np.max()))  
            t_energy = np.arange(len(energy)) * hop_length_energy / sr
            axs[current_ax].plot(t_energy, energy, color="red", alpha=0.7, label="Energy")
            axs[current_ax].legend(loc='upper right')
        axs[current_ax].set_title("Waveform")
        axs[current_ax].set_ylabel("Amplitude")
        axs[current_ax].set_xlim([0, max_time])
        axs[current_ax].grid(True, axis='x', linestyle='--', alpha=0.3)
        current_ax += 1
    
    if x is not None:
        x_np = x[sample_idx].detach().cpu().numpy()
        if x_np.shape[0] < x_np.shape[1]:
            x_np = x_np.T
        im = axs[current_ax].imshow(x_np.T, aspect="auto", origin="lower", cmap="magma", 
                                   extent=[0, x_np.shape[0]*hop_length/sr, 0, x_np.shape[1]])
        axs[current_ax].set_title("Spectrogram")
        axs[current_ax].set_ylabel("Mel Bin")
        axs[current_ax].set_xlim([0, max_time])
        axs[current_ax].grid(True, axis='x', linestyle='--', alpha=0.3)
        current_ax += 1
    
    if p is not None:
        p_np = p[sample_idx].detach().cpu().numpy()
        if p_np.ndim > 1:
            p_np = p_np.squeeze()
        t_p = np.arange(len(p_np)) * hop_length / sr
        axs[current_ax].plot(t_p, p_np, color="tab:green")
        axs[current_ax].set_title("Pitch")
        axs[current_ax].set_ylabel("Frequency (Hz)")
        axs[current_ax].set_xlim([0, max_time])
        axs[current_ax].grid(True, axis='both', linestyle='--', alpha=0.3)
        axs[current_ax].set_ylim([0, min(1000, p_np.max() * 1.2)])
        current_ax += 1
    
    if per is not None:
        per_np = per[sample_idx].detach().cpu().numpy()
        if per_np.ndim > 1:
            per_np = per_np.squeeze()
        t_per = np.arange(len(per_np)) * hop_length / sr
        axs[current_ax].plot(t_per, per_np, color="tab:red")
        axs[current_ax].set_title("Period (Voice Activity)")
        axs[current_ax].set_ylabel("periodocity")
        axs[current_ax].set_xlim([0, max_time])
        axs[current_ax].grid(True, axis='both', linestyle='--', alpha=0.3)
        axs[current_ax].set_ylim([-0.05, 1.05])
        axs[current_ax].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    
    if markers is not None:
        for i, t in enumerate(markers):
            label = marker_labels[i] if marker_labels and i < len(marker_labels) else None
            for ax in axs:
                ax.axvline(x=t, color='k', linestyle='-', alpha=0.7, label=label if i == 0 else None)
        if marker_labels:
            axs[0].legend(loc='upper right', fontsize='small')
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return fig

def dict_to(d, device, dtype=dtype):
    """Because PyTorch should have this built-in but doesn't"""
    return {k: v.to(device, dtype) if isinstance(v, torch.Tensor) else v 
            for k, v in d.items()}
    
def exists(v):
    return v is not None

def default(v, b):
    return v if exists(v) else b

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Conv2d(nn.Conv2d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias) -> Tensor:
        return super()._conv_forward(x, weight.to(x.device, x.dtype), None if bias is None else bias.to(x.device, x.dtype))

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
    
class RMSNorm(nn.Module):
    def __init__(self, dims: Union[int, Tensor, List, Tuple], 
                 eps = 1e-8, elementwise_affine = True):
        super(RMSNorm, self).__init__()
        if isinstance(dims, int):
            self.normalized_shape = (dims,)
        else:
            self.normalized_shape = tuple(dims)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
            init.ones_(self.weight)  
        else:
            self.register_parameter("weight", None)
    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
    
def LayerNorm(x: Tensor, normalized_shape: Union[int, Tensor, List, Tuple],
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
    return F.layer_norm(x, normalized_shape, weight, bias, eps)

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dtype():
    return torch.float32 if torch.cuda.is_available() else torch.float64

def tox():
    return {"device": get_device(), "dtype": get_dtype()}

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def align_f0(f0, ctx):
    b, l = f0.shape
    if l == ctx:
        return f0.squeeze(0).float()  
    frames_per_token = l / ctx
    idx = torch.arange(ctx, device=device, dtype=dtype)
    src_idx = (idx * frames_per_token).long().clamp(0, l-1)
    batch_idx = torch.arange(b, device=device, dtype=dtype).unsqueeze(1)
    f0 = f0[batch_idx, src_idx]
    return f0.squeeze(0).float()

def align_f0(f0, target_length, method='nearest', device=device, dtype=dtype):
    if device is None:
        device = f0.device
    if dtype is None:
        dtype = f0.dtype
    original_shape = f0.shape
    squeeze_batch = False
    reshape_back = None
    
    if f0.dim() == 1:
        f0 = f0.unsqueeze(0)
        squeeze_batch = True
    elif f0.dim() == 2:
        pass
    elif f0.dim() == 3:
        batch_size, seq_len, length = f0.shape
        f0 = f0.view(-1, length)
        reshape_back = (batch_size, seq_len)
    else:
        raise ValueError(f"F0 tensor must be 1D, 2D, or 3D, got {f0.dim()}D")
    batch_size, current_length = f0.shape
    if current_length == target_length:
        result = f0
    elif method == 'nearest':
        frames_per_token = current_length / target_length
        target_indices = torch.arange(target_length, device=device, dtype=torch.float32)
        source_indices = (target_indices * frames_per_token).long().clamp(0, current_length - 1)
        batch_indices = torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(1)
        result = f0[batch_indices, source_indices]
    else:
        import torch.nn.functional as F
        f0_for_interp = f0.unsqueeze(1)
        mode_map = {'linear': 'linear', 'cubic': 'bicubic'}
        if method not in mode_map:
            raise ValueError(f"Method '{method}' not supported. Use 'nearest', 'linear', or 'cubic'")
        
        result = F.interpolate(
            f0_for_interp.float(),
            size=target_length,
            mode=mode_map[method],
            align_corners=False
        ).squeeze(1)
    
    if reshape_back is not None:
        result = result.view(reshape_back[0], reshape_back[1], target_length)
    elif squeeze_batch:
        result = result.squeeze(0)
    return result.to(dtype)


class rotary(nn.Module):
    _seen = set()  
    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, radii=False,
                 learned_radius=False, learned_theta=False, learned_pitch=False, debug: List[str] = [], 
                 use_pbias=False, use_2d_axial=False, spec_shape=None):
        super().__init__()

        self.use_pbias = False
        self.use_2d_axial = use_2d_axial
        self.spec_shape = spec_shape
        self.last_f0_theta = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32 
        self.debug = debug
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.radii = radii
        f0_factor = 0.5
        self.learned_adaptation: bool = False
        pitch_scale = 1.0
        radius = 1
        
        if self.learned_adaptation:
            self.f0_scale = nn.Parameter(torch.tensor(f0_factor, device=self.device, dtype=self.dtype), requires_grad=True)
        else:
            self.register_buffer('f0_scale', torch.tensor(f0_factor))

        self.theta = nn.Parameter(torch.tensor(theta, device=self.device, dtype=self.dtype), requires_grad=True)
        self.pitch_scale = nn.Parameter(torch.tensor(pitch_scale, device=self.device, dtype=self.dtype), requires_grad=True)
        
        if use_2d_axial and spec_shape is not None:
            time_frames, freq_bins = spec_shape
            self.time_frames = time_frames
            self.freq_bins = freq_bins
            
            time_theta = 50.0
            time_freqs = 1.0 / (time_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('time_freqs', time_freqs)
            
            freq_theta = 100.0
            freq_freqs = 1.0 / (freq_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('freq_freqs', freq_freqs)
        else:
            freqs = 1. / (theta ** (torch.arange(0, dims, 2, device=self.device, dtype=self.dtype)[:(dims // 2)].float() / dims))
            self.freqs = nn.Parameter(torch.tensor(freqs, device=self.device, dtype=self.dtype), requires_grad=True)
        self.radius = nn.Parameter(torch.ones(radius, device=self.device, dtype=self.dtype), requires_grad=True)

    def compute_2d_axial_freqs(self, seq_len):
        if not self.use_2d_axial:
            return None
        time_frames = self.time_frames
        freq_bins = self.freq_bins
    
        t = torch.arange(seq_len, device=self.device, dtype=self.dtype)
        t_x = (t % time_frames).float()
        t_y = torch.div(t, time_frames, rounding_mode='floor').float()
        freqs_x = torch.outer(t_x, self.time_freqs)
        freqs_y = torch.outer(t_y, self.freq_freqs)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    def align_f0(self, f0, ctx):
        b, l = f0.shape
        if l == ctx:
            return f0.squeeze(0).float()  
        frames_per_token = l / ctx
        idx = torch.arange(ctx, device=self.device, dtype=self.dtype)
        src_idx = (idx * frames_per_token).long().clamp(0, l-1)
        batch_idx = torch.arange(b, device=self.device, dtype=self.dtype).unsqueeze(1)
        f0 = f0[batch_idx, src_idx]
        return f0.squeeze(0).float()

    def get_pitch_bias(self, f0):
        if f0 is None:
            return None
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)) * self.pitch_scale)
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def forward(self, x=None, f0=None, layer=None, input_type="audio") -> Tensor:
        if isinstance(x, int):
            ctx = x
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            batch, ctx, dims = x.shape
        else:
            batch, head, ctx, head_dim = x.shape
            
        if self.use_2d_axial and input_type == "spectrogram":
            freqs_2d = self.compute_2d_axial_freqs(ctx)
            if freqs_2d is not None:
                return freqs_2d.unsqueeze(0)
                
        t = torch.arange(ctx, device=self.device, dtype=self.dtype)

        if f0 is not None:
            f0_mean = f0.mean() + 1e-8
            theta = f0_mean * self.pitch_scale
            freqs = 1.0 / (theta ** (torch.arange(0, self.dims, 2, device=self.device, dtype=self.dtype)[:(self.dims // 2)].float() / self.dims))
            if "rotary" in self.debug:
                print(f"{layer}: {theta:.2f} : {f0_mean:.2f} : {ctx} ")
        else:
            freqs = self.freqs
            
        freqs = t[:, None] * freqs[None, :]
        
        if self.radii:
            if f0 is not None:
                radius = self.align_f0(f0, ctx)
            else:
                radius = self.radius
                if "rotary" in self.debug:
                    print(f"{layer} radius: {radius} ctx: {ctx}")
        else:
            radius = freqs
            
        freqs = torch.polar(torch.ones_like(radius), freqs.unsqueeze(0))
        
        self._counter += 1
        return freqs.unsqueeze(0)

    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        if x1.ndim == 2:
            x1 = x1.unsqueeze(0)
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        return torch.cat([x1.type_as(x), x2], dim=-1)

class MultiheadA(nn.Module):
    _seen = set()  
    rbf = False
    def __init__(self, dims: int, head: int, rotary_emb: bool = True, 
                 zero_val: float = 1e-4, minz: float = 1e-6, maxz: float = 1e-3, debug: List[str] = [], optim_attn=False):
        super(MultiheadA, self).__init__()

        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.debug = debug
        self._counter = 0

        self.q = Linear(dims, dims).to(device, dtype)
        self.k = Linear(dims, dims, bias=False).to(device, dtype)
        self.v = Linear(dims, dims).to(device, dtype)
        self.o = Linear(dims, dims).to(device, dtype)

        self.pad_token = 0
        self.rotary_emb = rotary_emb
        self.minz = minz
        self.maxz = maxz
        self.zero_val = zero_val
        self.optim_attn = optim_attn        
        self.fzero = nn.Parameter(torch.tensor(zero_val, device=device, dtype=dtype), requires_grad=False)
        
        if rotary_emb:
            self.rope_2d = rotary(
                dims=self.head_dim,
                use_2d_axial=False,
                spec_shape=(1500, 128),  
                debug=debug
            )
            self.rope = rotary(
                dims=self.head_dim,
                debug = debug,
                radii=False,
                learned_pitch=False,
                learned_freq=False,
                learned_theta=False,
                learned_radius=False,
                )
        else:
            self.rope_2d = None
            self.rope = None

    def enhanced_attention_scores(self, q, k, rbf_sigma=1.0, rbf_ratio=0.0):
        scale = (self.dims // self.head) ** -0.25
        dot_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if rbf_ratio <= 0.0:
            return dot_scores
        q_norm = q.pow(2).sum(dim=-1, keepdim=True)
        k_norm = k.pow(2).sum(dim=-1, keepdim=True)
        qk = torch.matmul(q, k.transpose(-1, -2))
        dist_sq = q_norm + k_norm.transpose(-1, -2) - 2 * qk
        rbf_scores = torch.exp(-dist_sq / (2 * rbf_sigma**2))
        return (1 - rbf_ratio) * dot_scores + rbf_ratio * rbf_scores
          
    def forward(self, x: Tensor, xa: Tensor = None, f0: Tensor = None, mask: Tensor = None, layer = None, feature_type="audio") -> tuple:
        x = x.to(device, dtype)
        if xa is not None:
            xa = xa.to(device, dtype)
            
        batch, ctx, dims = x.shape
        scale = (self.dims // self.head) ** -0.25
        
        z = default(xa, x).to(device, dtype)
        q = self.q(x)
        k = self.k(z)
        v = self.v(z)
        qlen = q.shape[1]
        klen = k.shape[1]

        if self.rotary_emb:   
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            qlen = q.shape[2]
            klen = k.shape[2]

            if feature_type == "spectrogram":
                input_type="spectrogram"
            else:
                input_type="audio"
            q = self.rope.apply_rotary(q, (self.rope(qlen, f0=f0, layer=layer, input_type=input_type)))
            k = self.rope.apply_rotary(k, (self.rope(klen, f0=f0, layer=layer, input_type=input_type)))
        else:
            q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
            batch, head, ctx, head_dim = q.shape
        
        if self.rbf:
            qk = self.enhanced_attention_scores(q * scale, k * scale, rbf_sigma=1.0, rbf_ratio=0.3)
        
        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        if f0 is not None and self.rope.use_pbias:
            pbias = self.rope.use_pbias(f0)
            if pbias is not None:
                qk = qk + pbias[:,:,:q.shape[2],:q.shape[2]]
        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), self.minz, self.maxz)
        zscale[token_ids.float() == self.pad_token] = fzero
        
        if mask is not None:
            mask = mask[:q.shape[2], :q.shape[2]]
            qk = qk + mask.unsqueeze(0).unsqueeze(0) * zscale.unsqueeze(-2).expand(qk.shape)
        qk = qk * zscale.unsqueeze(-2)
        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        
        if "multihead" in self.debug and self._counter % 100 == 0:
            print(f"MHA: q={q.shape}, k={k.shape}, v={v.shape} - {qk.shape}, wv shape: {wv.shape}")
        self._counter += 1        
        return self.o(wv), qk.detach()

class t_gate(nn.Module):
    def __init__(self, dims, num_types=4):
        super().__init__()
        self.gate_projections = nn.ModuleList([
            nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            for _ in range(num_types)])
        self.type_classifier = nn.Sequential(
            Linear(dims, num_types),
            nn.Softmax(dim=-1))
    def forward(self, x):
        type_probs = self.type_classifier(x)
        gates = torch.stack([gate(x) for gate in self.gate_projections], dim=-1)
        comb_gate = torch.sum(gates * type_probs.unsqueeze(2), dim=-1)
        return comb_gate

class m_gate(nn.Module):
    def __init__(self, dims, mem_size=64):
        super().__init__()
        self.m_key = nn.Parameter(torch.randn(mem_size, dims))
        self.m_val = nn.Parameter(torch.randn(mem_size, 1))
        self.gate_proj = nn.Sequential(Linear(dims, dims//2), nn.SiLU(), Linear(dims//2, 1))
        
    def forward(self, x):
        d_gate = torch.sigmoid(self.gate_proj(x))
        attention = torch.matmul(x, self.m_key.transpose(0, 1))
        attention = F.softmax(attention / math.sqrt(x.shape[-1]), dim=-1)
        m_gate = torch.matmul(attention, self.m_val)
        m_gate = torch.sigmoid(m_gate)
        return 0.5 * (d_gate + m_gate)

class c_gate(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.s_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.w_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.p_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.integ = Linear(dims*3, dims)
        
    def forward(self, x, features):
        s_feat = features.get("spectrogram", x)
        w_feat = features.get("waveform", x)
        p_feat = features.get("pitch", x)
        s = self.s_gate(x) * s_feat
        w = self.w_gate(x) * w_feat
        p = self.p_gate(x) * p_feat
        comb = torch.cat([s, w, p], dim=-1)
        return self.integ(comb)

class Residual(nn.Module):
    _seen = set()  
    def __init__(self, ctx, dims, head, act, cross_attn=True, debug: List[str] = [], 
                 tgate=True, mgate=False, cgate=False, mem_size=512, features=None):
        super().__init__()
        
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.cross_attn = cross_attn
        self.features = features
        self.debug = debug
        self._counter = 0
        self.dropout = 0.01
       
        self.t_gate = tgate
        self.m_gate = mgate
        self.c_gate = cgate
        
        self.blend = nn.Parameter(torch.tensor(0.5))
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), 
                  "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), 
                  "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), 
                  "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())

        self.attna = MultiheadA(dims, head, rotary_emb=True, debug=debug)
        self.attnb = (MultiheadA(dims, head, rotary_emb=True, debug=debug) if cross_attn else None)
        
        mlp = dims * 4
        self.mlp = nn.Sequential(Linear(dims, mlp), act_fn, Linear(mlp, dims))
        
        self.t_gate = t_gate(dims=dims, num_types=4) if t_gate else None
        self.m_gate = m_gate(dims=dims, mem_size=mem_size) if m_gate else None
        self.c_gate = c_gate(dims=dims) if c_gate else None
        
        self.lna = RMSNorm(dims)
        self.lnb = RMSNorm(dims) if cross_attn else None
        self.lnc = RMSNorm(dims)

        if not any([t_gate, m_gate, c_gate]):
            self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())

    def forward(self, x, xa=None, mask=None, f0=None, mode=None, layer=None, feature_type="audio") -> Tensor:
        x = x.to(device, dtype)
        if xa is not None:
            xa = xa.to(device, dtype)

        bln = self.blend
        x = x + self.attna(self.lna(x), xa=None, mask=mask, f0=f0, layer=layer, feature_type=feature_type)[0]
        
        if self.attnb and xa is not None:
            c = self.attnb(self.lnb(x), xa=xa, f0=f0, mask=None, layer=layer, feature_type=feature_type)[0]
            b = torch.sigmoid(bln)
            x = b * x + (1 - b) * c
        
        normx = self.lnc(x)
        mlp_out = self.mlp(normx)
        
        if self.t_gate:
            gate = self.t_gate(normx)
            x = x + gate * mlp_out
               
        elif self.m_gate:
            gate = self.m_gate(normx)
            x = x + gate * mlp_out
        
        elif self.c_gate and mode is not None:
            gate_output = self.c_gate(normx, self.features)
            x = x + gate_output

        else:
            if hasattr(self, 'mlp_gate'):
                mlp_gate = self.mlp_gate(normx)
                x = x + mlp_gate * mlp_out
            else:
                x = x + mlp_out
                
        if "residual" in self.debug and self._counter % 100 == 0:
            print(f"Step {self._counter}: Residual block output shape: {x.shape}, xa shape: {xa.shape if xa is not None else None}")        
            if self.t_gate:
                print(f"Step {self._counter}: Using t_gate: {self.t_gate}")
            elif self.m_gate:
                print(f"Step {self._counter}: Using m_gate: {self.m_gate}")
            elif self.c_gate:
                print(f"Step {self._counter}: Using c_gate: {self.c_gate}")
            else:
                print(f"Step {self._counter}: Using MLP gate: {self.mlp_gate if hasattr(self, 'mlp_gate') else None}")
        self._counter += 1      
                  
        return x

class FEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, stride=1, use_rope=False, spec_shape=None):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head  
        self.dropout = 0.01 
        self.use_rope = use_rope
        self.dims = dims
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims, kernel_size=kernel_size, stride=stride, padding=kernel_size//2), act_fn,
            Conv1d(dims, dims, kernel_size=5, padding=2), act_fn,
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims), act_fn)
        
        if use_rope:
            if spec_shape is not None:
                self.rope = rotary(
                    dims=self.head_dim,
                    use_2d_axial=True,
                    spec_shape=spec_shape, debug=[])
            else:
                self.rope = rotary(
                    dims=self.head_dim,
                    use_2d_axial=False, debug=[])
        else:
            self.rope = None
            self.positional = lambda length: sinusoids(length, dims)
            
        self.norm = RMSNorm(dims)
        self._norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, f0=None, layer=None, feature_type="audio"):
        if not self.use_rope or self.rope is None:
            return x
            
        batch, seq_len, dims = x.shape
        x = x.view(batch, seq_len, self.head, self.head_dim).permute(0, 2, 1, 3)
        if feature_type == "spectrogram" and hasattr(self.rope, 'use_2d_axial') and self.rope.use_2d_axial:
            rope_freqs = self.rope(seq_len, f0=f0, layer=layer, input_type="spectrogram")
        else:
            rope_freqs = self.rope(seq_len, f0=f0, layer=layer, input_type="audio")
        x = self.rope.apply_rotary(x, rope_freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, dims)
        return x

    def forward(self, x, f0=None, layer=None, feature_type="audio"):
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, f0=f0, layer=layer, feature_type=feature_type)
        else:
            x = x + self.positional(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self._norm(x)
        return x

class WEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, use_rope=False):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dropout = 0.01
        self.use_rope = use_rope
        self.dims = dims
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.downsample = nn.Sequential(
            Conv1d(input_dims, dims//8, kernel_size=15, stride=8, padding=7), act_fn,
            Conv1d(dims//8, dims//4, kernel_size=7, stride=4, padding=3), act_fn,
            Conv1d(dims//4, dims, kernel_size=9, stride=5, padding=4), act_fn)
        
        self.encoder = nn.Sequential(
            Conv1d(dims, dims, kernel_size=3, padding=1, groups=dims//8),  act_fn,
            Conv1d(dims, dims, kernel_size=1), act_fn)
        if use_rope:
            self.rope = rotary(
                dims=self.head_dim,
                use_2d_axial=False,
                theta=50.0, debug=[])
        else:
            self.rope = None
            self.positional = lambda length: sinusoids(length, dims)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, f0=None, layer=None):
        if not self.use_rope or self.rope is None:
            return x
        batch, seq_len, dims = x.shape
        x = x.view(batch, seq_len, self.head, self.head_dim).permute(0, 2, 1, 3)
        rope_freqs = self.rope(seq_len, f0=f0, layer=layer, input_type="waveform")
        x = self.rope.apply_rotary(x, rope_freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, dims)
        return x
        
    def forward(self, x, f0=None, layer=None, feature_type="waveform"):
        x = self.downsample(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, f0=f0, layer=layer)
        else:
            x = x + self.positional(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.norm(x)

class PEncoder(nn.Module):
    def __init__(self, input_dims, dims, head, layer, kernel_size, act, use_rope=False):
        super().__init__()
        
        self.head = head
        self.head_dim = dims // head
        self.dropout = 0.01
        self.use_rope = use_rope
        self.dims = dims
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        self.encoder = nn.Sequential(
            Conv1d(input_dims, dims//4, kernel_size=7, stride=8, padding=3), act_fn,
            Conv1d(dims//4, dims//2, kernel_size=5, stride=4, padding=2), act_fn,
            Conv1d(dims//2, dims, kernel_size=5, stride=5, padding=2), act_fn)
        
        if use_rope:
            self.rope = rotary(
                dims=self.head_dim,
                use_2d_axial=False,
                theta=100.0, debug=[])
        else:
            self.rope = None
            self.positional = lambda length: sinusoids(length, dims)
        self.norm = RMSNorm(dims)

    def apply_rope_to_features(self, x, f0=None, layer=None):
        if not self.use_rope or self.rope is None:
            return x
        batch, seq_len, dims = x.shape
        x = x.view(batch, seq_len, self.head, self.head_dim).permute(0, 2, 1, 3)
        rope_freqs = self.rope(seq_len, f0=f0, layer=layer, input_type="pitch")
        x = self.rope.apply_rotary(x, rope_freqs)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, dims)
        return x
        
    def forward(self, x, f0=None, layer=None, feature_type="pitch"):
        x = self.encoder(x).permute(0, 2, 1)
        if self.use_rope:
            x = self.apply_rope_to_features(x, f0=f0, layer=layer)
        else:
            x = x + self.positional(x.shape[1]).to(x.device, x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)
        return x

class AudioEncoder(nn.Module):
    _seen = set()  
    def __init__(self, mels: int, ctx: int, dims: int, head: int, layer: int, debug: List[str], features: List[str], 
                 f0_rotary: bool = False, act: str = "gelu"):
        super(AudioEncoder, self).__init__()

        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head
        self.debug = debug
        self._counter = 0

        self.features = features
        self.dropout = 0.01

        self.rope = rotary(
            dims=self.head_dim,
           )

        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "swish": nn.SiLU(), 
        "tanhshrink": nn.Tanhshrink(), "softplus": nn.Softplus(), "softshrink": nn.Softshrink(), "leaky_relu": nn.LeakyReLU(), "elu": nn.ELU()}
        act_fn = act_map.get(act, nn.GELU())
        
        if features == ["spectrogram", "waveform", "pitch"]:
            cgate=True
        else:
            cgate = False
            
        self.blocks = nn.ModuleDict({
            "spectrogram": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "spectrogram" in features else None
            ), 
            "waveform": nn.ModuleList(
            [WEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=11, act=act_fn)] +
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "waveform" in features else None
            ),
            "pitch": nn.ModuleList(
            [FEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=9, act=act, stride=2)] +
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug, features=features, cgate=cgate) for _ in range(layer)] if "pitch" in features else None
            ),       
            "spec_envelope": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug) for _ in range(layer)] if "spec_envelope" in features else None
            ),
            "spec_phase": nn.ModuleList(
            [FEncoder(input_dims=mels, dims=dims, head=head, layer=layer, kernel_size=3, act=act_fn)] + 
            [Residual(ctx=ctx, dims=dims, head=head, act=act, debug=debug) for _ in range(layer)] if "spec_phase" in features else None),
            })

        self.f0 = nn.ModuleList([
            FEncoder(input_dims=1, dims=dims, head=head, layer=layer, kernel_size=9, act=act, stride=2)
            for _ in range(layer)])

        self.rope_2d = rotary(
            dims=self.head_dim,
            use_2d_axial=True,
            spec_shape=(ctx, mels),
            debug=debug
        )
        
        self.rope_1d = rotary(
            dims=self.head_dim,
            use_2d_axial=False,
            debug=debug
        )

    def forward(self, enc, f0=None, layer="ENC"):
        enc = dict_to(enc, device, dtype)
        
        if self._counter < 1:
            s = enc.get("spectrogram")
            w = enc.get("waveform")
            p = f0 if f0 is not None else default(enc.get("pitch"), enc.get("f0"))
            plot_waveform(x=s, w=w, p=p, hop_length=128)

        out = {}
        out.update(enc)

        for f in self.features:
            if f in enc and f in self.blocks:
                x = enc[f]
                for block in self.blocks[f]:
                    x = block(x, f0=f0, layer=layer, feature_type=f)
                out[f] = x
                        
        if "encoder" in self.debug and self._counter % 100 == 0:
            names = list(x.keys()) 
            shapes = {k: v.shape for k, v in x.items()}
            print(f"Step {self._counter}: mode: {names}: shapes: {shapes}")
        self._counter += 1
        return out

class TextDecoder(nn.Module):
    def __init__(self, vocab: int, ctx: int, dims: int, head: int, layer: int, cross_attn: bool, 
                debug: List[str], features: List[str], f0_rotary: bool = False, sequential=False): 
        super(TextDecoder, self).__init__()
                
        self.dims = dims
        self.head = head
        self.ctx = ctx
        self.head_dim = dims // head

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.debug = debug
        self._counter = 0

        self.dropout = 0.01
        self.sequential = sequential
        self.features = features
        self.f0_rotary = f0_rotary
        
        self.token = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
        with torch.no_grad():
            self.token.weight[0].zero_()
        self.positional = nn.Parameter(data=torch.empty(ctx, dims), requires_grad=True)
        
        self.block = nn.ModuleList([
            Residual(ctx=ctx, dims=dims, head=head, act="gelu", cross_attn=cross_attn, debug=debug, features=features)
            for _ in range(layer)])
        
        self.blocks = nn.ModuleDict({
        f: nn.ModuleList([Residual(ctx=ctx, dims=dims, head=head, act="gelu", cross_attn=cross_attn, debug=debug, features=features)
            for _ in range(layer)]) for f in features})
        
        self.blend = nn.ParameterDict({f: nn.Parameter(torch.tensor(0.5)) for f in features})
        self.ln_dec = RMSNorm(dims)
        
        mask = torch.tril(torch.ones(ctx, ctx), diagonal=0)        
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, enc, f0=None, order=None, layer='DEC') -> Tensor:
        enc = dict_to(enc, device, dtype)
        x = x.to(device)
        bln = self.blend

        if order is None:
            order = self.features
        
        mask = self.mask[:x.shape[1], :x.shape[1]]
        x = self.token(x) + self.positional[:x.shape[1]]
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for block in self.block:
            x = block(x, xa=None, f0=f0, mask=mask, layer=layer)

        for f in order:
            if f in enc:
                xa = enc[f]
                for block in self.blocks[f]:
                    out = block(x=x, xa=xa, f0=f0, mask=None, layer=layer)

                a = torch.sigmoid(bln[f])
                x = a * out + (1 - a) * x
                        
        if "decoder" in self.debug and self._counter % 100 == 0:
            print(f"Step {self._counter}: Decoder output shape: {x.shape}, enc keys: {list(enc.keys())}, order: {order}")
        self._counter += 1  

        x = self.ln_dec(x)   
        return x @ torch.transpose(self.token.weight.to(dtype), 0, 1).float()

class Echo(nn.Module):
    def __init__(self, param: Dimensions):
        super().__init__()
        self.param = param
        self.count = 0

        self.encoder = AudioEncoder(
            mels=param.mels,
            ctx=param.aud_ctx,
            dims=param.aud_dims,
            head=param.aud_head,
            layer=param.aud_idx,
            act=param.act,
            debug=param.debug,
            features=param.features,
            f0_rotary=param.f0_rotary,
            )
        
        self.decoder = TextDecoder(
            vocab=param.vocab,
            ctx=param.text_ctx,
            dims=param.text_dims,
            head=param.text_head,
            layer=param.text_idx,
            cross_attn=param.cross_attn,
            debug=param.debug,
            features=param.features,
            f0_rotary=param.f0_rotary,
            )

        all_head = torch.zeros(self.param.text_idx, self.param.text_head, dtype=torch.bool)
        all_head[self.param.text_idx // 2 :] = True
        self.register_buffer("alignment_head", all_head.to_sparse(), persistent=False)

    def set_alignment_head(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        mask = torch.from_numpy(array).reshape(
            self.param.text_idx, self.param.text_head)
        self.register_buffer("alignment_head", mask.to_sparse(), persistent=False)

    def embed_audio(self, spectrogram: torch.Tensor):
        return self.encoder(spectrogram)

    def logits(self,input_ids: torch.Tensor, encoder_output: torch.Tensor):
        return self.decoder(input_ids, encoder_output)
        
    def forward(self,
        decoder_input_ids=None,
        labels=None,
        waveform: Optional[torch.Tensor]=None,
        input_ids=None,
        spectrogram: torch.Tensor=None,
        pitch: Optional[torch.Tensor]=None,
        f0: Optional[torch.Tensor]=None,
        f0d: Optional[torch.Tensor]=None,
        envelope: Optional[torch.Tensor]=None,
        phase: Optional[torch.Tensor]=None,
        ) -> Dict[str, torch.Tensor]:

        decoder_input_ids = input_ids
        encoder_inputs = {}
        if spectrogram is not None:
            encoder_inputs["spectrogram"] = spectrogram
        if waveform is not None:
            encoder_inputs["waveform"] = waveform
        if pitch is not None:
            encoder_inputs["pitch"] = pitch
        if envelope is not None:
            encoder_inputs["envelope"] = envelope
        if phase is not None:
            encoder_inputs["phase"] = phase
        if f0 is not None:
            encoder_inputs["f0"] = f0
        if f0d is not None:
            encoder_inputs["f0d"] = f0d
            
        encoder_outputs = self.encoder(encoder_inputs, f0=f0)
        logits = self.decoder(input_ids, encoder_outputs, f0=f0d)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=0)
                
        self.count += 1
        return {
            "logits": logits,
            "loss": loss,
            "labels": labels,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_outputs,
            } 

    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _init_weights(self, module):
        std = 0.02
        self.init_counts = {
            "Linear": 0, "Conv1d": 0, "LayerNorm": 0, "RMSNorm": 0,
            "Conv2d": 0, "SEBlock": 0, "TextDecoder": 0, "AudioEncoder": 0, 
            "Residual": 0, "MultiheadA": 0, "MultiheadB - Cross Attention": 0, 
            "MultiheadC": 0, "MultiheadD": 0, "FEncoder": 0,
            "WEncoder": 0, "PEncoder": 0}

        for name, module in self.named_modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
                self.init_counts["RMSNorm"] += 1
            elif isinstance(module, nn.Linear):
                if module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Linear"] += 1
            elif isinstance(module, Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv1d"] += 1
            elif isinstance(module, Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                self.init_counts["Conv2d"] += 1
            elif isinstance(module, MultiheadA):

                self.init_counts["MultiheadA"] += 1
            elif isinstance(module, TextDecoder):
                self.init_counts["TextDecoder"] += 1
            elif isinstance(module, AudioEncoder):
                self.init_counts["AudioEncoder"] += 1
            elif isinstance(module, Residual):
                self.init_counts["Residual"] += 1
    
    def init_weights(self):
        print("Initializing model weights...")
        self.apply(self._init_weights)
        print("Initialization summary:")
        for module_type, count in self.init_counts.items():
            if count > 0:
                print(f"{module_type}: {count}")

    def register_gradient_hooks(self):
        """Add this method to your Echo model class"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "encoder" in name:
                    param.register_hook(lambda grad, n=name: self._print_encoder_grad(n, grad))
                elif "decoder" in name:
                    param.register_hook(lambda grad, n=name: self._print_decoder_grad(n, grad))
        
        print("Gradient debugging hooks registered")
        return self

    def _print_encoder_grad(self, name, grad):
        if grad is not None and self.count == 10:  
            norm = grad.median().item()
            print(f"ENCODER GRAD: {name} = {norm:.6f}")
        
        return None

    def _print_decoder_grad(self, name, grad):
        if grad is not None and self.count == 10: 
            norm = grad.median().item()
            print(f"DECODER GRAD: {name} = {norm:.6f}")
        return None

    def reset_counter(self):
        """Reset the internal counter for debugging purposes."""
        self._counter = 0
        print("Counter reset to 0.")

metric = evaluate.load(path="wer")

def align_f0(f0, ctx):
    ctx = torch.tensor(ctx)
    bat, length = f0.shape
    if length == ctx:
        return f0 
    frames = length / ctx
    idx = torch.arange(ctx, device=f0.device)
    idx = (idx * frames).long()
    batch_idx = torch.arange(bat, device=f0.device).unsqueeze(1)
    return f0[batch_idx, idx.unsqueeze(0).expand(bat, -1)]
    
@dataclass
class DataCollator:
    tokenizer: Any
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1
        
        batch = {}

        if "spectrogram" in features[0] and features[0]["spectrogram"] is not None:
            spectrogram_list = [f["spectrogram"] for f in features]
            max_len_feat = max(f.shape[-1] for f in spectrogram_list)
            pad_spectrogram = []
            for feat in spectrogram_list:                
                current_len = feat.shape[-1]
                padding = max_len_feat - current_len
                if padding > 0:
                    pad_feat = F.pad(feat, (0, padding), mode='constant', value=pad_token_id)
                else:
                    pad_feat = feat
                pad_spectrogram.append(pad_feat)
            batch["spectrogram"] = torch.stack(pad_spectrogram)
            
        if "waveform" in features[0] and features[0]["waveform"] is not None:
            waveform_list = [f["waveform"] for f in features]
            max_len_wav = max(w.shape[-1] for w in waveform_list)
            pad_waveforms = []
            for wav in waveform_list:
                current_len = wav.shape[-1]
                padding = max_len_wav - current_len
                if padding > 0:
                    if wav.ndim == 1:
                        wav = wav.unsqueeze(0)
                    pad_wav = F.pad(wav, (0, padding), mode='constant', value=pad_token_id)
                else:
                    pad_wav = wav
                pad_waveforms.append(pad_wav)
            batch["waveform"] = torch.stack(pad_waveforms)

        if "label" in features[0] and features[0]["label"] is not None:
            labels_list = [f["label"] for f in features]
            max_len = max(len(l) for l in labels_list)
            all_ids = []
            all_labels = []

            for label in labels_list:
                label_list = label.tolist() if isinstance(label, torch.Tensor) else label
                decoder_input = [bos_token_id] + label_list                
                label_eos = label_list + [pad_token_id]  
                input_len = max_len + 1 - len(decoder_input)
                label_len = max_len + 1 - len(label_eos)                
                padded_input = decoder_input + [pad_token_id] * input_len
                padded_labels = label_eos + [pad_token_id] * label_len                
                all_ids.append(padded_input)
                all_labels.append(padded_labels)            
            batch["input_ids"] = torch.tensor(all_ids, dtype=torch.long)
            batch["labels"] = torch.tensor(all_labels, dtype=torch.long)

        if "pitch" in features[0] and features[0]["pitch"] is not None:
            pitch_list = [f["pitch"] for f in features]
            max_len_pitch = max(e.shape[-1] for e in pitch_list)
            pad_pitch = []
            for pitch in pitch_list:
                current_len = pitch.shape[-1]
                padding = max_len_pitch - current_len
                if padding > 0:
                    pad_pitch_item = F.pad(pitch, (0, padding), mode='constant', value=pad_token_id)
                else:
                    pad_pitch_item = pitch
                pad_pitch.append(pad_pitch_item)
            batch["pitch"] = torch.stack(pad_pitch)

        if "f0" in features[0] and features[0]["f0"] is not None:
            input_ids_batch = batch.get("input_ids", None)  
            if input_ids_batch is not None:
                target_length = input_ids_batch.shape[-1] 
                aligned_list = []
                original_list = []
                for feature in features:
                    f0 = feature["f0"]
                    original_list.append(f0)
                    if f0.shape[-1] != target_length:
                        aligned_f0 = align_f0(f0.unsqueeze(0), target_length).squeeze(0)
                    else:
                        aligned_f0 = f0
                    aligned_list.append(aligned_f0)
                batch["f0d"] = torch.stack(aligned_list)  
                batch["f0"] = torch.stack(original_list)  
        
        if "envelope" in features[0] and features[0]["envelope"] is not None:
            env_list = [f["envelope"] for f in features]
            max_len = max(f.shape[-1] for f in env_list)
            pad_env = []
            for feat in env_list:                
                current_len = feat.shape[-1]
                padding = max_len_feat - current_len
                if padding > 0:
                    pad_feat = F.pad(feat, (0, padding), mode='constant', value=pad_token_id)
                else:
                    pad_feat = feat
                pad_env.append(pad_feat)
            batch["envelope"] = torch.stack(pad_env)

        if "phase" in features[0] and features[0]["phase"] is not None:
            ph_list = [f["phase"] for f in features]
            max_len = max(f.shape[-1] for f in ph_list)
            pad_ph = []
            for feat in ph_list:                
                current_len = feat.shape[-1]
                padding = max_len_feat - current_len
                if padding > 0:
                    pad_feat = F.pad(feat, (0, padding), mode='constant', value=pad_token_id)
                else:
                    pad_feat = feat
                pad_ph.append(pad_feat)
            batch["phase"] = torch.stack(pad_ph) 

        return batch

def hilbert_transform(x):
    N = x.shape[-1]
    xf = torch.fft.rfft(x)
    h = torch.zeros(N // 2 + 1, device=x.device, dtype=x.dtype)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    return torch.fft.irfft(xf * h, n=N)

def analytic_signal(x):
    return x + 1j * hilbert_transform(x)

def hilbert_transform_2d(x, dim=-1):
    N = x.shape[dim]
    if dim == -1 or dim == len(x.shape) - 1:
        xf = torch.fft.rfft(x)
    else:
        xf = torch.fft.rfft(x, dim=dim)
    h_shape = [1] * len(x.shape)
    h_shape[dim] = N // 2 + 1
    h = torch.zeros(h_shape, device=x.device, dtype=x.dtype)
    if dim == -1 or dim == len(x.shape) - 1:
        if N % 2 == 0:
            h[..., 0] = h[..., -1] = 1
            h[..., 1:-1] = 2
        else:
            h[..., 0] = 1
            h[..., 1:] = 2
    else:
        pass
    return torch.fft.irfft(xf * h, n=N, dim=dim)

def hilbert_transform_true_2d(x):
    xf = torch.fft.rfft2(x)
    h1, h2 = torch.meshgrid(
        torch.fft.rfftfreq(x.shape[-2]) * 2 - 1,
        torch.fft.rfftfreq(x.shape[-1]) * 2 - 1,
        indexing='ij')
    h = -1j / (math.pi * (h1 + 1j*h2))
    h[0, 0] = 0 
    return torch.fft.irfft2(xf * h.to(x.device))

def process_spectrogram_with_hilbert(spec):
    analytic = spec + 1j * hilbert_transform(spec)
    envelope = torch.abs(analytic)
    phase = torch.angle(analytic)
    return envelope, phase
        
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
        
    return waveform.flatten()

def extract_features(batch, tokenizer, spectrogram, waveforms, pitch, frequency=False,
                     hop_length=128, fmin=0, fmax=8000, n_mels=128, n_fft=1024, sampling_rate=16000,
                     pad_mode="constant", center=True, power=2.0, window_fn=torch.hann_window, mel_scale="htk", 
                     norm=None, normalized=False, downsamples=False, period=False, hilbert=False):

    dtype = torch.float32
    device = torch.device("cuda:0")
    audio = batch["audio"]
    sampling_rate = audio["sampling_rate"]
    sr = audio["sampling_rate"]
    wav = load_wave(wave_data=audio, sample_rate=sr)

    if spectrogram:
        transform = torchaudio.transforms.MelSpectrogram(
            f_max=fmax,
            f_min=fmin,
            n_mels=n_mels,
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            norm=norm,
            normalized=normalized,
            power=power,
            center=center, 
            mel_scale=mel_scale,
            window_fn=window_fn,
            pad_mode=pad_mode)
        
        mel_spectrogram = transform(wav)      
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        spec = (log_mel + 4.0) / 4.0
        spec = torch.tensor(spec)
        batch["spectrogram"] = spec
        
    if hilbert:
        envelope_list = []
        phase_list = []
        
        for ch_idx in range(spec.shape[0]):
            envelope, phase = process_spectrogram_with_hilbert(spec[ch_idx])
            envelope_list.append(envelope)
            phase_list.append(phase)
            
        batch["envelope"] = torch.stack(envelope_list)
        batch["phase"] = torch.stack(phase_list)
        
    wav_1d = wav.unsqueeze(0)
    
    if waveforms:
        batch["waveform"] = wav_1d
            
    if pitch:
        wav_np = wav.numpy().astype(np.float64)  
        f0, t = pw.dio(wav_np, sampling_rate, 
                    frame_period=hop_length/sampling_rate*1000)
        f0 = pw.stonemask(wav_np, f0, t, sampling_rate)
        f0 = torch.from_numpy(f0).float()
        batch["pitch"] = f0.unsqueeze(0)  
        
    if frequency:
        wav_np = wav.numpy().astype(np.float64)  
        f0, t = pw.dio(wav_np, sampling_rate, frame_period=hop_length/sampling_rate*1000)
        f0 = pw.stonemask(wav_np, f0, t, sampling_rate)
        f0 = torch.from_numpy(f0).float()   
        batch["f0"] = f0
                  
    if spectrogram and waveforms and pitch:
        spec_mean = batch["spectrogram"].mean()
        spec_std = batch["spectrogram"].std() + 1e-6
        batch["spectrogram"] = (batch["spectrogram"] - spec_mean) / spec_std
        
        wav_mean = batch["waveform"].mean()
        wav_std = batch["waveform"].std() + 1e-6
        batch["waveform"] = (batch["waveform"] - wav_mean) / wav_std
        
        if batch["pitch"].max() > 1.0:
            pitch_min = 50.0
            pitch_max = 500.0
            batch["pitch"] = (batch["pitch"] - pitch_min) / (pitch_max - pitch_min)
            
    batch["label"] = tokenizer.encode(batch["transcription"], add_special_tokens=False)
    return batch

def compute_metrics(eval_pred, compute_result: bool = True, 
                    print_pred: bool = False, num_samples: int = 0, tokenizer=None, pitch=None, model=None):
    
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
            
    if hasattr(label_ids, "tolist"):
        label_ids = label_ids.tolist()
        
    label_ids = [[0 if token == -100 else token for token in seq] for seq in label_ids]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=False)

    if print_pred:
        for i in range(min(num_samples, len(pred_str))):
            print(f"Preds: {pred_str[i]}")
            print(f"Label: {label_str[i]}")
            print(f"preds: {pred_ids[i]}")
            print(f"label: {label_ids[i]}")
            print("--------------------------------")  
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    if model is None:
        global global_model
        if 'global_model' in globals():
            model = global_model
    
    if model is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
        if trainable_params > 0:
            efficiency_score = (100 - wer) / trainable_params
        else:
            print("Warning: Zero trainable parameters detected")
            efficiency_score = 0.0
    else:
        print("Warning: Model not available for parameter counting")
        trainable_params = 0.0
        efficiency_score = 0.0
    
    if hasattr(wer, "item"):
        wer = wer.item()
    
    metrics = {
        "wer": float(wer),
        "trainable_params_M": float(trainable_params),
        "efficiency_score": float(efficiency_score),
    }
    
    return metrics

logger = logging.getLogger(__name__)

def create_model(param: Dimensions) -> Echo:
    model = Echo(param).to('cuda')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    return model

def setup_tokenizer(token: str, local_tokenizer_path: str = "D:/newmodel/model/tokenn/"):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f"{local_tokenizer_path}/tokenizer.json")
    orig_encode = tokenizer.encode
    def enc(text, add_special_tokens=True):
        ids = orig_encode(text).ids
        if not add_special_tokens:
            sp_ids = [tokenizer.token_to_id(t) for t in ["<PAD>", "<BOS>", "<EOS>"]]
            ids = [id for id in ids if id not in sp_ids]
        return ids
    def bdec(ids_list, skip_special_tokens=True):
        results = []
        for ids in ids_list:
            if skip_special_tokens:
                ids = [id for id in ids if id not in [0, 1, 2]]
            results.append(tokenizer.decode(ids))
        return results
    def save_pretrained(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(f"{save_dir}/tokenizer.json")
    tokenizer.encode = enc
    tokenizer.batch_decode = bdec
    tokenizer.save_pretrained = save_pretrained
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    return tokenizer

def prepare_datasets(tokenizer, token: str, sanity_check: bool = False, dataset_config: Optional[Dict] = None) -> Tuple[any, any]:
    if dataset_config is None:
        dataset_config = {
            "spectrogram": True,
            "waveforms": True,
            "pitch": True,
            "frequency": True,
            "downsamples": True,
            "hop_length": 128,
            "fmin": 50,
            "fmax": 2000,
            "n_mels": 128,
            "n_fft": 1024,
            "sampling_rate": 16000,
        }
    
    dataset = load_dataset(
        "google/fleurs", 
        "en_us", 
        token=token, 
        trust_remote_code=True,
        streaming=False)
    
    dataset = dataset.cast_column(column="audio", feature=Audio(sampling_rate=16000)).select_columns(["audio", "transcription"])
    
    if sanity_check:
        dataset = dataset["test"].take(10)
        dataset = dataset.select_columns(["audio", "transcription"])
        logger.info(f"Sanity dataset size: {dataset.num_rows}") 
        print(f"Sanity dataset size: {dataset.num_rows}")
        prepare_fn = partial(extract_features, tokenizer=tokenizer, **dataset_config)
        
        dataset = dataset.map(
            function=prepare_fn, 
            remove_columns=["audio", "transcription"]
        ).with_format(type="torch")
        train_dataset = dataset
        test_dataset = dataset
    else:
        def filter_func(x):
            return (0 < len(x["transcription"]) < 512 and
                   len(x["audio"]["array"]) > 0 and
                   len(x["audio"]["array"]) < 1500 * 160)
        
        dataset = dataset.filter(filter_func).shuffle(seed=4)
        logger.info(f"Dataset size: {dataset['train'].num_rows}, {dataset['test'].num_rows}")
        print(f"Dataset size: {dataset['train'].num_rows}, {dataset['test'].num_rows}")
        prepare_fn = partial(extract_features, tokenizer=tokenizer, **dataset_config)
        columns_to_remove = list(next(iter(dataset.values())).features)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"].take(50)
        logger.info(f"Train dataset size: {train_dataset.num_rows}, Test dataset size: {test_dataset.num_rows}")
        
        train_dataset = train_dataset.map(
            function=prepare_fn, 
            remove_columns=columns_to_remove
        ).with_format(type="torch")
        
        test_dataset = test_dataset.map(
            function=prepare_fn, 
            remove_columns=columns_to_remove
        ).with_format(type="torch")
    
    return train_dataset, test_dataset

def get_training_args(
    log_dir: str,
    batch_eval_metrics: bool = False,
    max_steps: int = 10,
    save_steps: int = 1000,
    eval_steps: int = 1,
    warmup_steps: int = 0,
    num_train_epochs: int = 1,
    logging_steps: int = 1,
    eval_on_start: bool = False,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
) -> Seq2SeqTrainingArguments:

    return Seq2SeqTrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        logging_dir=log_dir,
        logging_strategy="steps",
        report_to=["tensorboard"],
        push_to_hub=False,
        disable_tqdm=False,
        save_total_limit=1,
        label_names=["labels"],
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_safetensors=False,
        eval_on_start=eval_on_start,
        batch_eval_metrics=batch_eval_metrics,
        max_grad_norm=max_grad_norm,     
    )

def main():
     
    token = ""
    log_dir = os.path.join('./output/logs', datetime.now().strftime(format='%m-%d_%H_%M_%S'))
    os.makedirs(name=log_dir, exist_ok=True)
    tokenizer = setup_tokenizer(token)

    def sanity(sanity: bool):

        if sanity:
            training_args = get_training_args(
            log_dir,
            batch_eval_metrics = False,
            max_steps = 10,
            save_steps = 0,
            eval_steps = 1,
            warmup_steps = 0,
            logging_steps = 1,
            eval_on_start = False,
            learning_rate = 5e-6,
            weight_decay = 0.01,
            )
        else:
            training_args = get_training_args(
            log_dir,
            batch_eval_metrics = False,
            max_steps = 1000,
            save_steps = 1005,
            eval_steps = 100,   
            warmup_steps = 100,
            logging_steps = 10,
            eval_on_start = False,
            learning_rate = 2.5e-4,
            weight_decay = 0.01,
            )

        return training_args
        
    param = Dimensions(
        mels=128,
        aud_ctx=1500,
        aud_head=4,
        aud_dims=512,
        aud_idx=4,
        vocab=40000,
        text_ctx=512,
        text_head=4,
        text_dims=512,
        text_idx=4,
        act="swish",
        debug={"rotary"},
        cross_attn=True,
        f0_rotary=False, 
        features = ["spectrogram"]
        )

    sanity_check = False
    training_args = sanity(sanity_check)
    dataset_config = {
        "spectrogram": True,
        "waveforms": False,
        "pitch": False,
        "downsamples": False,
        "frequency": False,
        "hilbert": False,
        "hop_length": 128,
        "fmin": 150,
        "fmax": 2000,
        "n_mels": 128,
        "n_fft": 1024,
        "sampling_rate": 16000,
        "pad_mode": "constant",
        "center": True, 
        "power": 2.0,
        "window_fn": torch.hann_window,
        "mel_scale": "htk",
        "norm": None,
        "normalized": False}
    
    model = create_model(param)
    
    global global_model
    global_model = model
    
    metrics_fn = partial(compute_metrics, print_pred=False, num_samples=5, 
                    tokenizer=tokenizer, model=model)
    
    print(f"{'Sanity check' if sanity_check else 'Training'} mode")
    train_dataset, test_dataset = prepare_datasets(
        tokenizer=tokenizer,
        token=token,
        sanity_check=sanity_check,
        dataset_config=dataset_config)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollator(tokenizer=tokenizer),
        compute_metrics=metrics_fn,
        ) 
       
    model.init_weights()
    trainer.train()

if __name__ == "__main__":
    main()

