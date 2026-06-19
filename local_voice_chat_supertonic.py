"""Bilingual (Arabic + English) local voice chat with selectable TTS.

Pipeline: STT (auto-detects the spoken language) -> gemma4:31b-cloud (replies in
that same language) -> TTS speaks it. Just talk in Arabic or English.

Two selectable TTS backends (--tts):
  * supertonic (default) https://github.com/supertone-inc/supertonic -- compact
                ONNX TTS, fast on CPU (RTF ~0.2), no GPU/sidecar/reference. Arabic
                is MSA-leaning. Voice from presets (M1-M5/F1-F5).
  * omnivoice   https://github.com/k2-fsa/OmniVoice -- diffusion TTS that clones a
                reference voice. Uses a Najdi/Saudi Arabic reference clip for
                Arabic and an English reference for English. Higher quality dialect
                but much slower on Mac (RTF ~2-4, ~140s to load).

Two selectable STT backends (--stt):
  * whisper  (default) faster-whisper "base", local on CPU, ~0.6s.
  * nemotron mlx-community/nemotron-3.5-asr-streaming-0.6b via mlx-audio (MLX).

    python local_voice_chat_supertonic.py                  # supertonic + whisper
    python local_voice_chat_supertonic.py --tts omnivoice  # Najdi voice cloning
    python local_voice_chat_supertonic.py --stt nemotron   # MLX nemotron STT
    python local_voice_chat_supertonic.py --voice-style F1 # supertonic speaker
"""
import re
import sys
import argparse
from pathlib import Path

import numpy as np
from fastrtc import ReplyOnPause, Stream
from loguru import logger
from ollama import chat

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


# --- text-to-speech backends -------------------------------------------------
TTS_BACKEND = "supertonic"  # set by argparse: "supertonic" | "omnivoice"
CHUNK_SAMPLES = 8192        # frames to the browser

# Supertonic
TOTAL_STEPS = 8             # flow-matching steps (quality/speed; 8 is the default)
SPEED = 1.05
VOICE_STYLE = "M1"          # set by argparse (M1-M5/F1-F5)
SUP_SR = 44100
_sup_tts = None
_sup_style = None

# OmniVoice -- reference clips for voice cloning (Najdi Arabic + English).
_TTS_BENCH = Path.home() / "Documents" / "tts-benchmark" / "data" / "refs"
OMNI_REFS = {
    "en": (_TTS_BENCH / "en_speaker.wav", "Some call me nature, others call me mother nature."),
    "ar": (_TTS_BENCH / "ar_speaker.wav",  # Najdi/Saudi dialect reference
           "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه."),
}
OMNI_SR = 24000
_omni = None


def init_tts() -> None:
    """Load the selected TTS backend (surfaces download/load cost up front)."""
    global _sup_tts, _sup_style, _omni
    if TTS_BACKEND == "supertonic":
        from supertonic import TTS
        _sup_tts = TTS(auto_download=True)
        _sup_style = _sup_tts.get_voice_style(voice_name=VOICE_STYLE)
    elif TTS_BACKEND == "omnivoice":
        import os
        import torch
        from omnivoice import OmniVoice
        # Metal (MPS) shares RAM with every other app, so OmniVoice OOMs mid-run
        # under memory pressure. CPU is slower (~RTF 5) but reliable, so it is the
        # default; set OMNIVOICE_DEVICE=mps to use the GPU when it has headroom.
        device = os.environ.get("OMNIVOICE_DEVICE", "cpu")
        _omni = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=torch.float32)
        logger.info(f"OmniVoice running on {device} (set OMNIVOICE_DEVICE=mps for GPU if it has free memory).")


def _sentences(text: str) -> list[str]:
    """Split into sentences so we synthesize and stream one at a time."""
    return [s.strip() for s in re.split(r"(?<=[.!?؟…])\s+", text.strip()) if s.strip()]


def tts_synthesize(text: str, lang: str):
    """Synthesize sentence-by-sentence with the active backend and yield
    (sample_rate, float32 frame) chunks so the first sentence plays while later
    ones generate."""
    for sentence in _sentences(text):
        if TTS_BACKEND == "omnivoice":
            ref_wav, ref_text = OMNI_REFS["ar" if lang == "ar" else "en"]
            out = _omni.generate(text=sentence, language=lang,
                                 ref_audio=str(ref_wav), ref_text=ref_text)
            audio = np.asarray(out[0], dtype=np.float32).reshape(-1)
            sr = OMNI_SR
        else:  # supertonic
            wav, _ = _sup_tts.synthesize(text=sentence, voice_style=_sup_style,
                                         total_steps=TOTAL_STEPS, speed=SPEED, lang=lang)
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            sr = SUP_SR
        for start in range(0, len(audio), CHUNK_SAMPLES):
            yield sr, audio[start:start + CHUNK_SAMPLES].reshape(1, -1)


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
    yield from tts_synthesize(response_text, voice_lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bilingual local voice chat (Supertonic / OmniVoice)")
    parser.add_argument("--tts", choices=["supertonic", "omnivoice"], default="supertonic",
                        help="text-to-speech backend: supertonic (fast, default) or "
                             "omnivoice (Najdi/Saudi Arabic voice cloning, slower)")
    parser.add_argument("--stt", choices=["whisper", "nemotron"], default="whisper",
                        help="speech-to-text backend (default: whisper)")
    parser.add_argument("--voice-style", default="M1",
                        help="Supertonic voice: M1-M5 or F1-F5 (default: M1; ignored for omnivoice)")
    args = parser.parse_args()
    STT_BACKEND = args.stt
    TTS_BACKEND = args.tts
    VOICE_STYLE = args.voice_style

    logger.info(f"Initializing STT ({STT_BACKEND}) and TTS ({TTS_BACKEND})...")
    init_stt()
    init_tts()
    logger.info(f"Launching voice chat (stt={STT_BACKEND}, tts={TTS_BACKEND})...")
    stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream.ui.launch()
