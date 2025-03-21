import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any
from torch import Tensor

class QueryModule(nn.Module):
    """Dedicated query projection module that handles only query transformations."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        assert dims % heads == 0, f"dims ({dims}) must be divisible by heads ({heads})"
        
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        self.scale = self.head_dim ** -0.25
        
        # Only query projection
        self.query = nn.Linear(in_features=dims, out_features=dims)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(tensor=self.query.weight, std=0.02)
        if self.query.bias is not None:
            nn.init.zeros_(tensor=self.query.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Project input to query representation
        
        Args:
            x: Input tensor [batch, seq_len, dims]
            
        Returns:
            Query tensor [batch, heads, seq_len, head_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Project and reshape for attention
        q = self.query(x)
        q = q.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply scaling pre-emptively for stable attention
        q = q * self.scale
        
        return q


class KeyModule(nn.Module):
    """Dedicated key projection module that handles only key transformations."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        assert dims % heads == 0, f"dims ({dims}) must be divisible by heads ({heads})"
        
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        self.scale = self.head_dim ** -0.25
        
        # Only key projection
        self.key = nn.Linear(in_features=dims, out_features=dims, bias=False)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(tensor=self.key.weight, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Project input to key representation
        
        Args:
            x: Input tensor [batch, seq_len, dims]
            
        Returns:
            Key tensor [batch, heads, seq_len, head_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Project and reshape for attention
        k = self.key(x)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply scaling pre-emptively for stable attention
        k = k * self.scale
        
        return k


class ValueModule(nn.Module):
    """Dedicated value projection module that handles only value transformations."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        assert dims % heads == 0, f"dims ({dims}) must be divisible by heads ({heads})"
        
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        
        # Only value projection
        self.value = nn.Linear(in_features=dims, out_features=dims)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(tensor=self.value.weight, std=0.02)
        if self.value.bias is not None:
            nn.init.zeros_(tensor=self.value.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Project input to value representation
        
        Args:
            x: Input tensor [batch, seq_len, dims]
            
        Returns:
            Value tensor [batch, heads, seq_len, head_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Project and reshape for attention
        v = self.value(x)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        return v


class KeyValueModule(nn.Module):
    """Dedicated key-value projection module that handles K and V transformations."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        # Use separate modules internally
        self.key_module = KeyModule(dims, heads)
        self.value_module = ValueModule(dims, heads)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Project input to key and value representations
        
        Args:
            x: Input tensor [batch, seq_len, dims]
            
        Returns:
            Tuple of (key, value) tensors, each shaped [batch, heads, seq_len, head_dim]
        """
        k = self.key_module(x)
        v = self.value_module(x)
        
        return k, v


class AttentionCombiner(nn.Module):
    """Combines separate Q and KV representations for attention computation."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        assert dims % heads == 0, f"dims ({dims}) must be divisible by heads ({heads})"
        
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        self.use_sdpa = True  # Use scaled dot product attention if available
        
        # Output projection
        self.out = nn.Linear(in_features=dims, out_features=dims)
        nn.init.normal_(tensor=self.out.weight, std=0.02)
        nn.init.zeros_(tensor=self.out.bias)
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute attention between provided q, k, v representations
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, key_len, head_dim]
            v: Value tensor [batch, heads, value_len, head_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, dims]
        """
        batch_size = q.size(0)
        seq_len = q.size(2)
        
        # Compute attention
        if self.use_sdpa:
            # Use PyTorch's optimized attention implementation if available
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                is_causal=(mask is not None and seq_len > 1)
            )
        else:
            # Manual implementation for older PyTorch versions
            # Note: No need for additional scaling here since we pre-scaled q and k
            attn = torch.matmul(q, k.transpose(-1, -2))
            
            if mask is not None:
                attn = attn + mask[:seq_len, :seq_len]
                
            attn = F.softmax(attn, dim=-1)
            attn_output = torch.matmul(attn, v)
        
        # Reshape and project output
        output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dims)
        return self.out(output)


class SeparatedAttention(nn.Module):
    """Full attention implementation with completely separated Q, K, and V modules."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        self.query_module = QueryModule(dims, heads)
        self.key_module = KeyModule(dims, heads)
        self.value_module = ValueModule(dims, heads)
        self.combiner = AttentionCombiner(dims, heads)
    
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through separated attention modules
        
        Args:
            x: Input tensor for query projection [batch, seq_len, dims]
            xa: Optional cross-attention tensor [batch, kv_len, dims]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, None)
        """
        # Project query from input sequence
        q = self.query_module(x)
        
        # Project keys and values from input or cross-attention input
        kv_input = xa if xa is not None else x
        k = self.key_module(kv_input)
        v = self.value_module(kv_input)
        
        # Compute attention and return
        output = self.combiner(q, k, v, mask)
        
        # Return attention weights for later use if needed (None for now)
        return output, None


# Example usage with MHA integration
class MultiHeadAttentionWithSeparation(nn.Module):
    """Demonstrates how to use SeparatedAttention in larger architecture."""
    
    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.attention = SeparatedAttention(dims, heads)
        self.layer_norm = nn.LayerNorm(dims)
    
    def forward(self, x, xa=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, xa, mask)
        return x + residual
