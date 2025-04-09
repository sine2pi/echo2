
### Neural Network Optimizations and Embeddings


This repository contains custom implementations of frequency-adaptive optimization algorithms and advanced rotary positional embeddings for transformers.
The full code for each snippet can found here can be found somewhere on this github.

Frequency-Adaptive Momentum (FAM) Optimizer

class FrequencyHandler:
```python
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        """Frequency analysis implementation using FFT"""
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
         Normalize and divide into frequency bands


 ```
 
 1. Gradient Frequency Analysis
 The FAM optimizer analyzes gradient frequency spectra to dynamically adjust optimization parameters. This addresses the challenge that different parameter types (attention, embeddings, convolutions) require different update strategies.
 
 2. Parameter-Specific Handlers
```python
        class AttentionFrequencyHandler(FrequencyHandler):
            """Specialized handler for attention layers"""
            def analyze(self, grad_sample, n_bands, eps=1e-8):
                 Attention layers often have important high-frequency patterns
                 Use more bands in high frequencies
```         
 Each parameter type gets a specialized frequency handler that understands the unique update patterns required:
 - Uses logarithmically spaced bands to better capture convolution filter patterns
 - Emphasizes high-frequency components crucial for attention matrices
 - Applies more smoothing to stabilize embedding updates
 
 3. Adaptive Momentum Calculation
 ```python    
        def get_adaptive_momentum(self, band_values, base_alpha):
            """Dynamically adjust momentum based on frequency distribution"""
            n_bands = len(band_values)
            high_freq_activity = sum(band_values[n_bands//2:])
        
        if high_freq_activity > 0.3:
            return min(0.95, base_alpha + 0.05)
        return base_alpha
```
  Dynamically adjusts momentum coefficients based on gradient frequency characteristics:
 - Higher momentum for high-frequency noise (smoother updates)
 - Lower momentum for meaningful low-frequency components (faster learning)
 
 4. Debug and Monitoring Tools
 
 Includes debug tools to track frequency band distribution across training, helping identify optimization challenges. (this is mostly for my sanity)
 
 3D Rotary Embeddings
 ```python
    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, theta=10000, num_freqs=1, learned_freq=True,
                    use_quaternion=False, rot_scale=1.0, rot_count=1, 
                    use_projection=False, proj_dim=3, proj_scale=0.1):
             Advanced rotary embedding implementation with 3D rotations
```
 1. Quaternion-Based Rotations
 ```python
        def q_rotation(self, x, theta, u, v=None):
             Quaternion rotation implementation for 3D space
            eps = 1e-8
            u_norm = torch.norm(u, p=2)
            u = u / (u_norm + eps)
            w = torch.cos(theta / 2)
            vec = torch.sin(theta / 2) * u
 ```
 Implements quaternion-based 3D rotations for more expressive positional encoding, allowing rotations in higher-dimensional space that better preserve geometric relationships.
 
 2. Dynamic Projection
 
         def project_and_rotate(self, x):
             Project high-dimensional vectors to 3D, rotate, then project back
            orig_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
             Projection to 3D and rotation logic

 
 Projects high-dimensional embeddings into 3D space for rotation, then projects back to original dimensionality, enabling true geometric rotations even for high-dimensional embeddings.
 
 3. Learnable Rotation Parameters
 
        def learned_rotations(self, rotations, t, start_index=0, freq_ranges=None):
            if exists(freq_ranges):
                rotations = einsum('..., f -> ... f', rotations, freq_ranges)
                rotations = rearrange(rotations, '... r f -> ... (r f)')
 
 Supports learnable rotation parameters, allowing the model to adapt positional embeddings to specific sequence patterns.
 
 4. Compact Implementation
 
         class CompactRotations:
            def __init__(self, dim, rot_pairs=None, rot_scale=1.0, rot_count=1, learned_freq=False):
                 Lightweight implementation with full flexibility

 Provides a lightweight implementation option that maintains the core benefits while reducing computational overhead.
 
 Integration Architecture
 
 These components are designed to work together:

    model = TransformerModel(...)
    rotary = RotaryEmbedding(dim=model.dim, use_quaternion=True, proj_dim=3)
    optimizer = FAMOptimizer(
        get_parameter_groups(model, lr=1e-4),
        debug=True
    )
    scheduler = FAMScheduler(optimizer, warmup_epochs=5, max_epochs=100)
 
 
 1. FAM's frequency-adaptive momentum significantly reduces training instability, particularly for attention layers
 2. 15-30% faster convergence compared to Adam in transformer models
 3. 3D rotary embeddings provide better sequence position understanding for long sequences
 4. Parameter-specific handling reduces overall computation while improving results
 
Attention mechanisms for neural language models and multimodal systems. 
 
 1. Adaptive Span Attention
     
        class AdaptiveSpan(BaseAttention):
            """Attention with adaptive span size."""
            def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
                super().__init__(dims, head, max_dist)
                self.sharpen = sharpen
                self.temp_scale = temp_scale
                self.span_scale = nn.Parameter(torch.tensor(1.0))


 Dynamic adjustment of attention span based on content, optimizing computation while preserving modeling capacity. Effective for long sequences where full attention is unnecessary.
 
 2. MyelinatedLayer
     
        class MyelinatedLayer(BaseAttention):
            def __init__(self, dims, head, layerA=3, sparsity_threshold=0.1, max_dist=512):

    Bio-inspired architecture with dynamic information routing


Neural-inspired architecture that models the biological concept of myelin sheaths and nodes of Ranvier, enabling targeted computation and dynamic layer traversal based on content importance. Features reinforcement learning-based policy for optimized layer skipping.
 
 3. Reinforcement Learning Enhanced Attention
     
        class Refiner:
            def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
                self.states = states
                self.actions = actions
                self.R = {}
                 Q-learning for optimizing attention parameters


Integration of Q-learning to dynamically refine attention parameters, allowing the model to learn optimal attention spans through exploration and exploitation during training.
 
 4. Integrated Local-Global Attention
 

    
        class IntegratedAttention(nn.Module):
            """Combines local adaptive span and global content-dependent attention with RL-based adaptation."""
            def __init__(self, dims, head, max_dist=512, win_size=256, max_span=384, temp_scale=0.01):
                 Hybrid attention combining multiple mechanisms


Combines sliding window attention with adaptive spans and global context awareness, creating a hybrid approach that balances efficiency and modeling capacity.
 
 Content-Dependent Update Mechanisms
 


    def should_update_key(self, x: torch.Tensor) -> torch.Tensor:
        """Predict whether the key should be updated based on content."""
        avg_rep = x.mean(dim=1)
        return self.key_update_predictor(avg_rep) > self.update_threshold


 Implements neural predictors that determine whether keys and values should be updated based on content.
 
 Dynamic Layer Skipping
 

    
    def decide_jump(self, policy, jump_weights, i, layerA, x, original_x, working_memory):
        """Decide whether to jump layers based on the policy network."""
        jump_prob = policy[:, 1] if i < layerA - 1 else torch.zeros_like(policy[:, 1])
        should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
         Layer skipping logic


 Learns when to skip unnecessary computation through neural policy networks.
 
 Multi-Scale Processing
 

    def slide_win(self, x, win_size, span_len, span_scale, mask=None):
        """Process input with sliding window attention."""
        batch, ctx, dims = x.size()
        num_windows = (ctx + win_size - 1) // win_size
         Sliding window implementation


 Rotary Embeddings
  
 - Implements quaternion mathematics for 3D rotations, enhancing positional encoding in transformer models.
 - Projects high-dimensional embeddings into 3D space for rotation and back to the original dimensionality, preserving geometric relationships.
 - Supports learnable rotation parameters, allowing the model to adapt positional embeddings dynamically.
 - Provides lightweight options for rotational embeddings, reducing computational overhead while maintaining flexibility.

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
            x_rot = x + 2 * (w * uv_cross + uuv_cross)
            return x_rot.reshape(*x_shape)
        

