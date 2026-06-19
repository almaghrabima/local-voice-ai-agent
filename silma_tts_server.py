"""silma TTS sidecar for the local voice agent.

Runs in the tts-benchmark venv (which already has f5_tts + torch installed):

    ~/Documents/tts-benchmark/.venvs/tts/bin/python silma_tts_server.py

silma-ai/silma-tts is a standard F5-TTS v1 checkpoint, so we load it directly
through f5_tts (same approach as tts-benchmark/src/generate_silma.py). The model
and both reference voices are loaded once at startup; each request just runs
inference. Exposes a tiny stdlib HTTP API so the agent (in a *different* venv)
can call it without sharing dependencies:

    GET  /health        -> 200 "ok" once the model is loaded
    POST /tts           -> 24 kHz mono WAV bytes
         body: {"text": "...", "voice": "en"|"ar"}
"""
from __future__ import annotations

import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import soundfile as sf

# --- paths into the tts-benchmark project (where the weights + refs live) -----
TTS_BENCH = Path.home() / "Documents" / "tts-benchmark"
CKPT = TTS_BENCH / "models" / "silma" / "model.pt"
VOCAB = TTS_BENCH / "models" / "silma" / "vocab.txt"
# Reference voices: prefer this project's refs/ (trimmed ~2.5s clips, which cut
# per-reply latency roughly in half), and fall back to the benchmark's full
# clips for any voice we haven't trimmed locally.
LOCAL_REFS = Path(__file__).resolve().parent / "refs"
BENCH_REFS = TTS_BENCH / "data" / "refs"

# DiT config from silma-ai/silma-tts config.yaml (== F5TTS_v1_Base arch)
SILMA_CFG = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)

HOST = "127.0.0.1"
PORT = int(os.environ.get("SILMA_PORT", "8001"))
NFE_STEP = int(os.environ.get("SILMA_NFE_STEP", "32"))
SAMPLE_RATE = 24000

# Populated by load() before the server starts accepting requests.
_MODEL = None
_VOCODER = None
_REFS: dict[str, tuple] = {}  # voice -> (ref_audio, ref_text)
_INFER = None


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load() -> None:
    """Load the silma checkpoint, vocoder, and both reference voices once."""
    global _MODEL, _VOCODER, _INFER, _DEVICE

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if not CKPT.exists():
        sys.exit(f"Missing {CKPT}. Run tts-benchmark/scripts/setup.sh to download weights.")

    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import (
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
        infer_process,
    )

    _DEVICE = _pick_device()
    print(f"[silma-tts] device={_DEVICE} nfe_step={NFE_STEP} loading model...", flush=True)
    _MODEL = load_model(DiT, SILMA_CFG, str(CKPT), vocab_file=str(VOCAB), device=_DEVICE)
    _VOCODER = load_vocoder()
    _INFER = infer_process

    for voice in ("en", "ar"):
        wav = LOCAL_REFS / f"{voice}_speaker.wav"
        if not wav.exists():
            wav = BENCH_REFS / f"{voice}_speaker.wav"
        txt = wav.with_suffix(".txt")
        if not wav.exists():
            print(f"[silma-tts] WARNING: reference {wav} missing; voice '{voice}' disabled", flush=True)
            continue
        ref_text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        _REFS[voice] = preprocess_ref_audio_text(str(wav), ref_text)
        secs = round(len(open(str(wav), "rb").read()) / 2 / 24000, 1)  # rough 16-bit/24k estimate
        print(f"[silma-tts] loaded voice '{voice}' from {wav.parent.name}/ (~{secs}s ref)", flush=True)

    print(f"[silma-tts] ready on http://{HOST}:{PORT}  voices={list(_REFS)}", flush=True)


def synthesize(text: str, voice: str, nfe_step: int, speed: float,
               cfg_strength: float = 2.0) -> bytes:
    """Run silma inference and return 24 kHz mono 16-bit WAV bytes."""
    if voice not in _REFS:
        voice = "en" if "en" in _REFS else next(iter(_REFS))
    ref_audio, ref_text = _REFS[voice]
    wave, sr, _ = _INFER(
        ref_audio, ref_text, text, _MODEL, _VOCODER,
        nfe_step=nfe_step, speed=speed, cfg_strength=cfg_strength, device=_DEVICE,
    )
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quieter logs
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/tts":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = (payload.get("text") or "").strip()
            voice = payload.get("voice", "en")
            nfe_step = int(payload.get("nfe_step", NFE_STEP))
            speed = float(payload.get("speed", 1.0))
            cfg_strength = float(payload.get("cfg_strength", 2.0))
            if not text:
                self._send(400, b"empty text", "text/plain")
                return
            wav = synthesize(text, voice, nfe_step, speed, cfg_strength)
            self._send(200, wav, "audio/wav")
        except Exception as exc:  # surface errors to the agent instead of hanging
            print(f"[silma-tts] error: {exc}", flush=True)
            self._send(500, str(exc).encode(), "text/plain")

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    load()
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
