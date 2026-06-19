# Local Voice AI Agent

A real-time voice chat application powered by local AI models. This project allows you to have voice conversations with AI models like Gemma running locally on your machine.

## Features

- Real-time speech-to-text conversion
- Local LLM inference using Ollama
- Text-to-speech response generation
- Web interface for interaction
- Phone number interface option

## Prerequisites

- MacOS
- [Ollama](https://ollama.ai/) - Run LLMs locally
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

## Installation

### 1. Install prerequisites with Homebrew

```bash
brew install ollama
brew install uv
```

### 2. Clone the repository

```bash
git clone https://github.com/jesuscopado/local-voice-ai-agent.git
cd local-voice-ai-agent
```

### 3. Set up Python environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 4. Download required models in Ollama

The scripts currently use `gemma4:31b-cloud` (an Ollama Cloud model — replies
leave your machine). Pull whichever model you want to run and set it in the
script, e.g.:

```bash
ollama pull gemma4:31b-cloud
# or a fully-local model:
ollama pull gemma3:4b
```

## Usage

### Basic Voice Chat

```bash
python local_voice_chat.py
```

### Advanced Voice Chat (with system prompt)

#### Web UI (default)
```bash
python local_voice_chat_advanced.py
```

#### Phone Number Interface
Get a temporary phone number that anyone can call to interact with your AI:
```bash
python local_voice_chat_advanced.py --phone
```

This will provide you with a temporary phone number that you can call to interact with the AI using your voice.

### Bilingual Voice Chat (Arabic + English, recommended)

`local_voice_chat_supertonic.py` is the recommended bilingual variant: speak
**Arabic or English** and it auto-detects the language, replies in that same
language, and speaks it back. It has two selectable TTS backends (`--tts`):

| TTS backend | Engine | Speed (M2) | Arabic | Notes |
| --- | --- | --- | --- | --- |
| `supertonic` (default) | [Supertonic](https://github.com/supertone-inc/supertonic), ONNX/CPU | **RTF ~0.2 (fast)** | MSA-leaning | No GPU, no reference voice; preset voices `M1`–`M5`/`F1`–`F5` |
| `omnivoice` | [OmniVoice](https://github.com/k2-fsa/OmniVoice), PyTorch | RTF ~5 CPU / ~2–4 MPS (slow) | **Najdi/Saudi** via voice cloning | Clones a Najdi reference clip; needs `omnivoice` installed |

```bash
python local_voice_chat_supertonic.py                  # supertonic + whisper (default)
python local_voice_chat_supertonic.py --tts omnivoice  # Najdi/Saudi Arabic (slow)
python local_voice_chat_supertonic.py --stt nemotron   # local MLX nemotron STT
python local_voice_chat_supertonic.py --voice-style F1  # supertonic voice: M1-M5 / F1-F5
```

**STT backends** are shared with the silma variant (`--stt whisper|nemotron`, see
the table below); whisper runs on the CPU so it doesn't compete for the GPU.

**OmniVoice notes:** install it with `uv pip install omnivoice` (pulls PyTorch).
It clones Najdi Arabic from `tts-benchmark/data/refs/ar_speaker.wav` and English
from `en_speaker.wav`. It runs on **CPU by default** (reliable but ~RTF 5, so
several seconds per reply); set `OMNIVOICE_DEVICE=mps` to use the GPU when it has
free memory (faster, but it OOMs under memory pressure since Metal shares RAM).

### Bilingual Voice Chat with silma TTS (Arabic + English)

`local_voice_chat_silma.py` is a bilingual variant: speak **Arabic or English**
and it auto-detects the language, replies in that same language, and speaks the
reply with the [silma-ai/silma-tts](https://huggingface.co/silma-ai/silma-tts)
voice that matches.

Instead of Kokoro, it streams speech from silma (an F5-TTS model). Because
silma needs heavy `f5_tts` / `torch` dependencies, it runs as a small HTTP
**sidecar** in its own environment, while the voice agent stays in this
project's `.venv`.

**Additional prerequisites:**

- The silma weights (`model.pt`, `vocab.txt`) and reference voices, plus a
  Python env with `f5_tts` installed. This setup lives in a sibling
  `~/Documents/tts-benchmark` project; the sidecar reads the weights from there
  and the trimmed English reference voice from this repo's `refs/`. Adjust the
  paths at the top of `silma_tts_server.py` if yours differ.
- `faster-whisper` (already in this project's dependencies) for multilingual
  speech-to-text.

**Run it (sidecar + agent together):**

```bash
./run_silma.sh                 # auto voice, whisper STT
./run_silma.sh ar              # force the Arabic reference voice
./run_silma.sh auto nemotron   # auto voice, local MLX nemotron STT
```

Or start the two processes manually:

```bash
# 1) silma TTS sidecar (in the env that has f5_tts)
~/Documents/tts-benchmark/.venvs/tts/bin/python silma_tts_server.py

# 2) the voice agent (in this project's .venv)
python local_voice_chat_silma.py
```

**Speech-to-text backends (`--stt`):**

| Backend | Where it runs | Notes |
| --- | --- | --- |
| `whisper` (default) | local, CPU | `faster-whisper base`, ~0.6s, Arabic + English |
| `nemotron` | local, Apple Silicon (MLX) | [`mlx-community/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b) via [`mlx-audio`](https://github.com/Blaizzy/mlx-audio), ~0.2s warm, Arabic + English. The model (~1.3 GB) downloads from Hugging Face on first run. |

`nemotron` needs `mlx-audio` (not on PyPI yet — install from git):

```bash
uv pip install "git+https://github.com/Blaizzy/mlx-audio.git"
python local_voice_chat_silma.py --stt nemotron   # local MLX nemotron ASR
python local_voice_chat_silma.py --voice ar         # force Arabic voice
```

**Tuning notes:** speech-to-first-audio is kept low (~2s) via a trimmed ~2.5s
reference clip, `nfe_step=12`, `speed=1.15`, and sentence/first-phrase
streaming — see the constants at the top of `local_voice_chat_silma.py` and
`silma_tts_server.py`.

## How it works

The application uses:
- `FastRTC` for WebRTC communication
- `Moonshine` for local speech-to-text conversion
- `Kokoro` for text-to-speech synthesis
- `Ollama` for running local LLM inference with `Gemma` models

When you speak, your audio is:
1. Transcribed to text using Moonshine
2. Sent to a local LLM via Ollama for processing
3. The LLM response is converted back to speech with Kokoro
4. The audio response is streamed back to you via FastRTC
