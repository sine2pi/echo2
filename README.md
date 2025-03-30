# Echo: Advanced Attention Speech Recognition

Echo is a neural speech recognition model that implements innovative attention mechanisms for efficient audio transcription. This architecture features n-dimensional quaternion rotations and reinforcement learning attention adaptations.

## Key Features

- **Myelinated Attention**: Biologically-inspired dynamic attention path selection with skip connections and working memory integration
- **n-dimensional Quaternion Rotary Embeddings**: Advanced positional encoding using quaternion mathematics for better sequence understanding
- **Q-learning Based Refiner**: Reinforcement learning for adaptive attention span optimization
- **Adaptive Focus Sliding Window**: Content-dependent windowing that dynamically allocates computation to important signal regions
- **Node Importance Tracking**: Dynamic computational routing based on token relevance scores
- **Integrated Attention**: Combines local and global attention with quality-based learning loops

## Usage

```python
from modelc import Echo, Dimensions, process_audio
from transformers import WhisperTokenizer

# Initialize tokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")

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
    cross_attention_type="myelinated", 
    
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
```

## Technical Details

Echo combines conventional transformer architecture with innovative attention mechanisms. The model leverages quaternion mathematics to implement rotational embeddings, Q-learning for adaptive attention spans, and working memory integration for long-range dependencies. The sliding window mechanism with adaptive focus enables efficient processing of audio signals by concentrating computation on the most informative segments.

## Requirements

- PyTorch 2.0+
- Transformers
- TorchAudio
- Datasets (HuggingFace)
- NumPy
- Einops

## License

MIT
