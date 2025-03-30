### Echo: Advanced Attention Speech Recognition - Work in progress

Echo is a neural speech recognition model that implements innovative attention mechanisms for efficient audio transcription. This architecture features n-dimensional quaternion rotations and reinforcement learning attention adaptations. The model leverages quaternion mathematics to implement rotational embeddings, Q-learning for adaptive attention spans, and working memory integration for long-range dependencies. The sliding window mechanism with adaptive focus enables efficient processing of audio signals by concentrating computation on the most informative segments.

### Key Features

- **Myelinated Attention**: Biologically-inspired dynamic attention path selection with skip connections and working memory integration
- **n-dimensional Quaternion Rotary Embeddings**: Advanced positional encoding using quaternion mathematics for better sequence understanding
- **Q-learning Based Refiner**: Reinforcement learning for adaptive attention span optimization
- **Adaptive Focus Sliding Window**: Content-dependent windowing that dynamically allocates computation to important signal regions
- **Node Importance Tracking**: Dynamic computational routing based on token relevance scores
- **Integrated Attention**: Combines local and global attention with quality-based learning loops



#### Usage

```python

# Configure model parameters
param = Dimensions(
    # Audio encoder parameters
    mels=80, audio_ctx=1500, audio_dims=512, audio_head=2,
    audio_layerA=2, audio_layerB=1, audio_act="gelu",
    
    # Text decoder parameters
    vocab=len(tokenizer), text_ctx=448, text_dims=512, 
    text_head=2, text_layerA=2, text_layerB=0, text_act="gelu",
    
    # Attention mechanism selection
    self_attention_type="myelinated",  # Options: "myelinated", "integrated", "adaptive"
    cross_attention_type="myelinated", # Options: "myelinated", "integrated", "adaptive"
    
    # Other parameters
    cross_attention=False,
    decoder_start_token_id=50258,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Initialize model
model = Echo(param=param).to('cuda')

# Process audio and generate transcription
audio_file = "audio_sample.wav"
mel_spectrogram = process_audio(audio_file, audio_ctx=param.audio_ctx, 
                               mels=param.mels, hop_length=160, n_fft=400, sr=16000)
encoded_audio = model.encoder(mel_spectrogram)

# Generate text
input_ids = torch.tensor([[param.decoder_start_token_id]]).to('cuda')
logits = model.decoder(input_ids, encoded_audio)

# Decode transcription
transcription = tokenizer.decode(torch.argmax(logits, dim=-1)[0])

        self.rotary1 = RotaryEmbedding(
            dim=dims//head,
            theta=10000,
            use_quaternion=True,
            use_projection=True, <- if you need more than 3 dimensions (else givens for 2Dn) 
            rot_scale=4.0,
            rot_count=1
        )

        self.rotary2 = RotaryEmbedding(
            dim=dims//head,
            theta=-6000, <-change direction
            use_quaternion=False, <- falls back to regular RoPE
            use_projection=False,
            rot_scale=1.0,
            rot_count=4
        )

      q = self.rotary1.rotate_queries_or_keys(q)
      k = self.rotary2.rotate_queries_or_keys(k)
      ... as many as you need
        see code for more options

```
 use_projection = True
 
    The code projects to a new 3D dimension using a learned linear projection
    Applies true quaternion rotations in this 3D space
    Projects back to the original dimension using another learned projection
    The projections are initialized as pseudo-inverses of each other

 use_projection = False
 
    Directly applies rotations in the original dimension
    For dimensions > 3, it falls back to using multiple 2D Givens rotations
    It does NOT use quaternion rotation except when exactly in 3D
    Each pair of dimensions gets rotated separately in 2D planes






## Requirements

- PyTorch 2.0+
- Transformers
- TorchAudio
- Datasets (HuggingFace)
- NumPy
- Einops

## License

MIT
