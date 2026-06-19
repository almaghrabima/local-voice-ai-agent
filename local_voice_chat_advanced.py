import sys
import argparse

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat

stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def echo(audio):
    transcript = stt_model.stt(audio)
    logger.debug(f"🎤 Transcript: {transcript}")
    response = chat(
        model="gemma4:e4b-mlx",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include emojis or special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
            {"role": "user", "content": transcript},
        ],
    )
    response_text = response["message"]["content"].strip()
    if not response_text:
        # Guard: some models occasionally return an empty completion, which
        # would make the agent play no audio and appear unresponsive.
        response_text = "Sorry, I didn't catch that. Could you say it again?"
    logger.debug(f"🤖 Response: {response_text}")
    for audio_chunk in tts_model.stream_tts_sync(response_text):
        yield audio_chunk


def create_stream():
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat Advanced")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    args = parser.parse_args()

    stream = create_stream()

    if args.phone:
        logger.info("Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("Launching with Gradio UI...")
        stream.ui.launch()
