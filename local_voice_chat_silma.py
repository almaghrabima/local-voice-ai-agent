"""Bilingual (Arabic + English) local voice chat with silma TTS.

Pipeline: STT (auto-detects the spoken language) -> gemma4:31b-cloud (replies in
that same language) -> silma sidecar (silma_tts_server.py) speaks it with the
matching reference voice. Just talk in Arabic or English and it follows.

Two selectable STT backends (--stt):
  * whisper  (default) faster-whisper "base", local on CPU, ~0.6s.
  * nemotron mlx-community/nemotron-3.5-asr-streaming-0.6b via mlx-audio. Runs
             locally on Apple Silicon (MLX), ~0.2s warm, Arabic + English.

Start the sidecar first (or use run_silma.sh which does both):

    ~/Documents/tts-benchmark/.venvs/tts/bin/python silma_tts_server.py
    python local_voice_chat_silma.py                  # whisper STT, auto voice
    python local_voice_chat_silma.py --stt nemotron   # local MLX nemotron STT
    python local_voice_chat_silma.py --voice ar        # force Arabic voice
"""
import io
import re
import sys
import argparse

import numpy as np
import requests
import soundfile as sf
from fastrtc import ReplyOnPause, Stream
from loguru import logger
from ollama import chat

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# --- speech-to-text backends -------------------------------------------------
STT_BACKEND = "whisper"  # set by argparse: "whisper" | "nemotron"
# MLX port of NVIDIA's nemotron ASR -- runs natively on Apple Silicon.
NEMOTRON_MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
_whisper = None
_nemotron = None


def init_stt() -> None:
    """Load/prepare the selected STT backend (and surface errors early)."""
    global _whisper, _nemotron
    if STT_BACKEND == "whisper":
        from faster_whisper import WhisperModel
        # "base" is the speed/quality sweet spot here: ~0.6s, correct on both
        # Arabic and English, and it reports the detected language.
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
    """Pick voice language from the transcript script (nemotron returns no
    language field). Any Arabic-range codepoint -> Arabic, else English."""
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

SILMA_URL = "http://127.0.0.1:8001/tts"
CHUNK_SAMPLES = 4096  # pace audio to the browser in ~0.17s frames

# silma speed knobs (see benchmarks): nfe_step=12 with sway sampling stays
# crisp; speed>1 shortens audio a touch. Paired with a trimmed ~2.5s reference
# voice in refs/, a short first phrase generates in ~0.9s.
NFE_STEP = 12
SPEED = 1.15
FIRST_CHUNK_MAX = 32  # chars: keep the first spoken unit short for fast first audio
MIN_CHUNK_BYTES = 12  # below ~10 bytes silma forces speed=0.3, which is slower


def _split_first(first: str) -> list[str]:
    """Break a long opening sentence into [head, tail] so the first spoken unit
    is short. Prefer a comma boundary near FIRST_CHUNK_MAX, else a word
    boundary. Returns [first] unchanged when it is already short."""
    if len(first) <= FIRST_CHUNK_MAX:
        return [first]
    window = first[:FIRST_CHUNK_MAX]
    cut = window.rfind(",")          # nicest: clause boundary
    keep_comma = cut != -1
    if cut < MIN_CHUNK_BYTES:        # no usable comma -> last word boundary
        cut = window.rfind(" ")
        keep_comma = False
    if cut < MIN_CHUNK_BYTES:        # one very long word -> leave as-is
        return [first]
    head = first[:cut + (1 if keep_comma else 0)].strip()
    tail = first[cut + 1:].strip()
    return [head, tail] if tail else [head]


# Voice override: "auto" picks the silma voice from the detected language;
# "en"/"ar" force one. Set by argparse in __main__.
VOICE = "auto"


def speakable_chunks(text: str) -> list[str]:
    """Break a reply into units we synthesize and stream one at a time.

    The first unit is kept short (the opening sentence is split at a comma or
    word boundary near FIRST_CHUNK_MAX) so the user hears audio in ~1-2s; later
    units stay sentence-sized for natural prosody. Tiny fragments are merged to
    avoid silma's <10-byte slow path."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?؟…])\s+", text.strip()) if s.strip()]
    if not sentences:
        return []
    chunks = _split_first(sentences[0]) + sentences[1:]
    # merge any too-short fragment into the next one
    merged: list[str] = []
    for c in chunks:
        if c and len(c.encode("utf-8")) < MIN_CHUNK_BYTES and merged:
            merged[-1] = f"{merged[-1]} {c}".strip()
        elif c:
            merged.append(c)
    return merged


def silma_tts(text: str, voice: str):
    """Synthesize sentence-by-sentence via the sidecar, yielding (sr, frame)
    chunks. The first sentence starts playing while later ones are still
    generating, so the user hears audio in ~1-2s instead of after the whole
    reply is rendered."""
    for sentence in speakable_chunks(text):
        try:
            resp = requests.post(
                SILMA_URL,
                json={"text": sentence, "voice": voice, "nfe_step": NFE_STEP, "speed": SPEED},
                timeout=120,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"silma sidecar request failed: {exc} (is silma_tts_server.py running?)")
            return
        audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if audio.ndim > 1:  # collapse to mono
            audio = audio.mean(axis=1)
        for start in range(0, len(audio), CHUNK_SAMPLES):
            yield sr, audio[start:start + CHUNK_SAMPLES].reshape(1, -1)


def echo(audio):
    transcript, lang = transcribe(audio)
    logger.debug(f"🎤 [{lang}] Transcript: {transcript}")
    if not transcript:
        return  # nothing intelligible heard; stay silent rather than hallucinate
    voice = VOICE if VOICE in ("en", "ar") else ("ar" if lang == "ar" else "en")
    response = chat(
        model="gemma4:31b-cloud",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful voice assistant in a live call. Reply in the SAME language the user spoke (Arabic or English). Keep it to one or two short, natural sentences. Your reply is read aloud, so use plain conversational text only: no markdown, lists, bullet points, emojis, or special characters.",
            },
            {"role": "user", "content": transcript},
        ],
    )
    response_text = response["message"]["content"].strip()
    if not response_text:
        # Guard: some models occasionally return an empty completion, which
        # would make the agent play no audio and appear unresponsive.
        response_text = "عذرا لم افهم. هل يمكنك الاعادة؟" if voice == "ar" else \
            "Sorry, I didn't catch that. Could you say it again?"
    logger.debug(f"🤖 [{voice}] Response: {response_text}")
    yield from silma_tts(response_text, voice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bilingual local voice chat with silma TTS")
    parser.add_argument("--voice", choices=["auto", "en", "ar"], default="auto",
                        help="silma voice: auto = follow detected language (default)")
    parser.add_argument("--stt", choices=["whisper", "nemotron"], default="whisper",
                        help="speech-to-text backend: whisper (faster-whisper, default) "
                             "or nemotron (local MLX nemotron via mlx-audio, ~0.2s warm)")
    args = parser.parse_args()
    VOICE = args.voice
    STT_BACKEND = args.stt

    logger.info(f"Initializing STT backend: {STT_BACKEND}...")
    init_stt()
    logger.info(f"Launching bilingual silma voice chat (voice={VOICE}, stt={STT_BACKEND})...")
    stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream.ui.launch()
