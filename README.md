# Echo: Adaptive Attention Speech-to-Text Model

Echo is a speech recognition model that implements multiple innovative attention mechanisms including Myelinated Attention, Integrated Attention, and Adaptive Update Attention.

## Features

- **Multiple attention mechanisms**: Choose between standard, myelinated, integrated, or adaptive attention types
- **Layer specialization**: Configure different attention mechanisms for early (LayerA) and deep (LayerB) transformer layers
- **Dynamic computation**: Myelinated attention adaptively focuses computation on important parts of sequences
- **Flexible architecture**: Easily customize model dimensions, context windows, and layer counts

## Usage

```python
import torch
from transformers import WhisperTokenizer

# Initialize tokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")

# Configure model parameters
param = Dimensions(
    # Audio encoder config
    mels=80,
    audio_ctx=1500,
    audio_head=2,
    audio_layerA=2,
    audio_layerB=1,
    audio_dims=512,
    audio_act="gelu",
    
    # Text decoder config
    vocab=len(tokenizer),
    text_ctx=448,
    text_head=2,
    text_layerA=2,
    text_layerB=0,
    text_dims=512,
    text_act="gelu",
    
    # Attention mechanism selection
    self_attention_type="myelinated",  # Options: "myelinated", "integrated", "adaptive"
    cross_attention_type="myelinated", 
    
    # Other parameters
    decoder_start_token_id=50258,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Initialize model
model = Echo(param=param).to('cuda')

# Process audio
mel_spectrogram = process_audio(audio_file, audio_ctx=param.audio_ctx, 
                               mels=param.mels, hop_length=160, n_fft=400, sr=16000)
encoded_audio = model.encoder(mel_spectrogram)

# Generate transcription
input_ids = torch.tensor([[param.decoder_start_token_id]]).to('cuda')
logits = model.decoder(input_ids, encoded_audio)
```

## Training

The model includes a comprehensive training pipeline with:
- Learning rate warmup and scheduling
- Mixed precision training
- TensorBoard logging
- Word Error Rate (WER) evaluation
- Gradient accumulation and clipping

## Requirements

- PyTorch 2.0+
- Transformers
- TorchAudio
- Datasets (HuggingFace)
- Evaluate (HuggingFace)

## License

MIT
