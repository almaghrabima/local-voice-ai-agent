"""Bilingual (Arabic + English) local voice chat with Supertonic TTS.

Pipeline: STT (auto-detects the spoken language) -> gemma4:31b-cloud (replies in
that same language) -> Supertonic speaks it. Just talk in Arabic or English.

Supertonic (https://github.com/supertone-inc/supertonic) is a compact ONNX TTS
that runs fast on the CPU -- no GPU, no sidecar, no reference voice, and it takes
the language per call. That keeps the GPU free and removes the silma sidecar.

Two selectable STT backends (--stt):
  * whisper  (default) faster-whisper "base", local on CPU, ~0.6s.
  * nemotron mlx-community/nemotron-3.5-asr-streaming-0.6b via mlx-audio (MLX).

    python local_voice_chat_supertonic.py                 # whisper STT
    python local_voice_chat_supertonic.py --stt nemotron  # MLX nemotron STT
    python local_voice_chat_supertonic.py --voice-style F1 # different speaker
"""
import re
import sys
import argparse

import numpy as np
from fastrtc import ReplyOnPause, Stream
from loguru import logger
from ollama import chat
from supertonic import TTS

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# --- speech-to-text backends -------------------------------------------------
STT_BACKEND = "whisper"  # set by argparse: "whisper" | "nemotron"
NEMOTRON_MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
_whisper = None
_nemotron = None


def init_stt() -> None:
    """Load/prepare the selected STT backend (and surface errors early)."""
    global _whisper, _nemotron
    if STT_BACKEND == "whisper":
        from faster_whisper import WhisperModel
        _whisper = WhisperModel("base", device="cpu", compute_type="int8")
        list(_whisper.transcribe(np.zeros(16000, dtype=np.float32))[0])  # warm up
    elif STT_BACKEND == "nemotron":
        import mlx.core as mx
        from mlx_audio.stt import load
        _nemotron = load(NEMOTRON_MODEL)
        _nemotron.generate(mx.array(np.zeros(16000, dtype=np.float32)))  # warm/compile


def _to_16k_mono(audio) -> np.ndarray:
    """fastrtc (sample_rate, int16 ndarray) -> 16 kHz mono float32."""
    sr, arr = audio
    a = np.asarray(arr).reshape(-1).astype(np.float32) / 32768.0
    if sr != 16000:
        n = round(len(a) * 16000 / sr)
        a = np.interp(np.linspace(0, len(a), n, endpoint=False),
                      np.arange(len(a)), a).astype(np.float32)
    return a


def _detect_lang(text: str) -> str:
    """Arabic script -> 'ar', else 'en' (nemotron returns no language field)."""
    return "ar" if any("؀" <= c <= "ۿ" for c in text) else "en"


def transcribe(audio) -> tuple[str, str]:
    """Return (text, language) from a fastrtc audio chunk."""
    a = _to_16k_mono(audio)
    if STT_BACKEND == "nemotron":
        import mlx.core as mx
        text = (_nemotron.generate(mx.array(a)).text or "").strip()
        return text, _detect_lang(text)
    segments, info = _whisper.transcribe(a, beam_size=1)
    text = "".join(s.text for s in segments).strip()
    return text, info.language


# --- text-to-speech (Supertonic) ---------------------------------------------
SAMPLE_RATE = 44100      # Supertonic output rate
TOTAL_STEPS = 8          # flow-matching steps (quality/speed; 8 is the default)
SPEED = 1.05
CHUNK_SAMPLES = 8192     # ~0.19s frames to the browser
VOICE_STYLE = "M1"       # set by argparse
_tts = None
_style = None


def init_tts() -> None:
    global _tts, _style
    _tts = TTS(auto_download=True)
    _style = _tts.get_voice_style(voice_name=VOICE_STYLE)


def _sentences(text: str) -> list[str]:
    """Split into sentences so we synthesize and stream one at a time."""
    return [s.strip() for s in re.split(r"(?<=[.!?؟…])\s+", text.strip()) if s.strip()]


def supertonic_tts(text: str, lang: str):
    """Synthesize sentence-by-sentence and yield (sample_rate, float32 frame)
    chunks, so the first sentence plays while later ones generate."""
    for sentence in _sentences(text):
        wav, _ = _tts.synthesize(text=sentence, voice_style=_style,
                                 total_steps=TOTAL_STEPS, speed=SPEED, lang=lang)
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        for start in range(0, len(audio), CHUNK_SAMPLES):
            yield SAMPLE_RATE, audio[start:start + CHUNK_SAMPLES].reshape(1, -1)


def echo(audio):
    transcript, lang = transcribe(audio)
    logger.debug(f"🎤 [{lang}] Transcript: {transcript}")
    if not transcript:
        return  # nothing intelligible heard; stay silent rather than hallucinate
    voice_lang = "ar" if lang == "ar" else "en"
    response = chat(
        model="gemma4:31b-cloud",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful voice assistant in a live call. Reply in the SAME language the user spoke (Arabic or English). Keep it to one or two short, natural sentences. Your reply is read aloud, so use plain conversational prose only: no markdown, lists, bullet points, or emojis.",
            },
            {"role": "user", "content": transcript},
        ],
    )
    response_text = response["message"]["content"].strip()
    if not response_text:
        response_text = "عذرا لم افهم. هل يمكنك الاعادة؟" if voice_lang == "ar" else \
            "Sorry, I didn't catch that. Could you say it again?"
    logger.debug(f"🤖 [{voice_lang}] Response: {response_text}")
    yield from supertonic_tts(response_text, voice_lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bilingual local voice chat with Supertonic TTS")
    parser.add_argument("--stt", choices=["whisper", "nemotron"], default="whisper",
                        help="speech-to-text backend (default: whisper)")
    parser.add_argument("--voice-style", default="M1",
                        help="Supertonic voice: M1-M5 or F1-F5 (default: M1)")
    args = parser.parse_args()
    STT_BACKEND = args.stt
    VOICE_STYLE = args.voice_style

    logger.info(f"Initializing STT ({STT_BACKEND}) and Supertonic TTS (voice {VOICE_STYLE})...")
    init_stt()
    init_tts()
    logger.info(f"Launching Supertonic voice chat (stt={STT_BACKEND}, voice={VOICE_STYLE})...")
    stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream.ui.launch()
