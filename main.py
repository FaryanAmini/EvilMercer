import os
import sys

# App deps live in ./app_libs (pip --target) so they never occupy Hermes
# pm-managed interpreter. Must precede every third-party import below.
APP_LIBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_libs")
if os.path.isdir(APP_LIBS) and APP_LIBS not in sys.path:
    sys.path.insert(0, APP_LIBS)

import threading
import time
import numpy as np
import pyaudio
from openwakeword.model import Model
from openwakeword.utils import AudioFeatures
from moonshine_voice import MicTranscriber, ModelArch, LineTextChanged, LineCompleted

import openwakeword


def _load_dotenv(path):
    """Minimal .env loader: KEY=value lines, quotes stripped, no clobber of real env vars."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Hermes lives in the gitignored ./hermes-agent checkout (not pip-installable)
HERMES_CHECKOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hermes-agent")
if HERMES_CHECKOUT not in sys.path:
    sys.path.insert(0, HERMES_CHECKOUT)
from run_agent import AIAgent

# Hermes brain endpoint (Qwen cloud, OpenAI-compatible /v1 wire)
# Key lives in .env as API_KEY; a real HERMES_API_KEY env var still wins.
HERMES_BASE_URL = os.getenv("HERMES_BASE_URL", "https://maas.qwencloudapi.com/compatible-mode/v1")
HERMES_API_KEY = os.getenv("HERMES_API_KEY") or os.getenv("API_KEY", "not-needed")
HERMES_MODEL = os.getenv("HERMES_MODEL", "qwen3.8-flash")

hermes_agent = AIAgent(
    base_url=HERMES_BASE_URL,
    api_key=HERMES_API_KEY,
    model=HERMES_MODEL,
    api_mode="chat_completions",  # plain OpenAI wire; skips Hermes' provider URL sniffing
    quiet_mode=True,
    request_overrides={"extra_body": {"enable_thinking": False}},

)

#  Qwen TTS via the dashscope SDK
# --- Local Qwen3-TTS via tts_server.py (own .venv-tts + GPU; see that file).
# Start the server BEFORE launching the assistant; if it's down, speak() just
# prints an error and the text-only flow keeps working.
LOCAL_TTS_URL = os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:8055/tts")

def speak(text):
    """Synthesize text via the local TTS server and block until playback ends."""
    if not text:
        return
    try:
        import io
        import json
        import urllib.request
        import wave
        import sounddevice as sd

        req = urllib.request.Request(
            LOCAL_TTS_URL,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        wav_bytes = urllib.request.urlopen(req, timeout=120).read()

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if channels == 2:
            samples = samples.reshape(-1, 2)
        sd.play(samples, samplerate=rate)
        sd.wait()  # don't return until the speakers go quiet -- mic cleanup depends on this
    except Exception as e:
        print(f"[TTS error: {type(e).__name__}: {e}]")


# Written for the ear: the reply is fed straight into TTS, so any markdown,
# bullets, code blocks, or URLs would be read out literally.
SYSTEM_PROMPT = (
    "You are a voice assistant. Your replies are converted to speech, so write "
    "for the ear: plain spoken sentences only. Never use markdown, bullet lists, "
    "code blocks, emojis, or URLs. Say numbers and abbreviations the way they "
    "should be read aloud. Keep answers to one or two short sentences unless the "
    "user asks for more. if asked to display things do it on the external monitor, not the laptop displays."
)

wakeModel = Model(wakeword_model_paths=[openwakeword.models["alexa"]["model_path"]])

FORMAT = pyaudio.PyAudio() if False else pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280  # Determines the frame step sizing

audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# Moonshine transcriber for post wake-word voice assistant logic
transcriber = MicTranscriber().model_arch(ModelArch.TINY)
transcriber.load()  # session-scoped printing lives in run_assistant_session's listener

# Rolling conversation memory: run_conversation takes history in and returns the
# updated transcript in result["messages"] -- nothing persists between calls
# unless we carry it ourselves.
chat_history = []

def drain_mic():
    """Discard whatever the mic captured while we were busy talking/thinking
    (stale backlog + the assistant's own TTS voice) and wipe the wake model's
    rolling feature history so none of it can retrigger detection."""
    time.sleep(0.3)
    backlog = stream.get_read_available()
    if backlog > 0:
        stream.read(backlog, exception_on_overflow=False)
    wakeModel.preprocessor = AudioFeatures()
    wakeModel.reset()

def run_assistant_session():
    """Transcribe one utterance. Returns the final text, or None if interrupted."""
    print("[*] Listening... (say something, Ctrl+C to cancel)")

    done = threading.Event()          # set by the LineCompleted listener
    session = {"final_text": None}    # mutable box to smuggle text back

    def session_listener(event):
        if isinstance(event, LineTextChanged):
            # live partials: print in place so the user sees the text as it firms up.
            # the :<60 pads so leftovers from a longer earlier partial get erased.
            text = event.line.text
            print(f"\r{text:<60}", end="", flush=True)

        elif isinstance(event, LineCompleted):
            # End of utterance: stash the final text, print it on a clean line,
            # then unblock the session. Without done.set() this session never returns.
            session["final_text"] = event.line.text
            print(f"\r{'':<60}\r[Finalized]: {event.line.text}")
            done.set()


    # Free the microphone so MicTranscriber can open its own input stream
    stream.stop_stream()
    try:
        transcriber.start()
        transcriber.push_listener(session_listener)

        # wait() a timeout (say 15s) so a session where you say nothing
        # can't hang forever. Check done.is_set() or use the return value.
        if not done.wait(timeout=15):
            print("\r[!] Timed out waiting for an utterance.")


    except KeyboardInterrupt:
        pass
    finally:
        transcriber.pop_listener()
        try:
            transcriber.stop()
        except Exception:
            pass
        stream.start_stream()
        drain_mic()

    return session["final_text"]

def main():
    try:
        while True:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            # Convert raw bytes to a 16-bit numpy array
            audio_frame = np.frombuffer(raw_data, dtype=np.int16)
            # Feed the frame into the model
            # openWakeWord handles internal buffering automatically

            prediction = wakeModel.predict(
                audio_frame,
                patience={"alexa_v0.1": 4},
                threshold={"alexa_v0.1": 0.5},
            )
            # Adjust threshold (e.g., 0.5) to fine-tune sensitivity
            if prediction["alexa_v0.1"] > 0.5:
                print("\n[!] Wake Word Detected! Activation triggered.")

                # Hand off to Moonshine for transcription of the user's command.
                # Agent runs here on the main thread -- never in the transcriber's
                # audio callback, which would block the mic pipeline.
                utterance = run_assistant_session()
                if utterance:
                    response = hermes_agent.run_conversation(
                        user_message=utterance,
                        # Only seed the prompt on the first turn; afterwards it
                        # rides along inside chat_history (passing it every turn
                        # would stack duplicate system messages).
                        system_message=SYSTEM_PROMPT if not chat_history else None,
                        conversation_history=chat_history,
                        task_id="launch",
                    )
                    # Carry the updated transcript into the next turn -- this is
                    # what gives the agent memory.
                    chat_history[:] = response["messages"]
                    reply = response["final_response"]
                    print(f"[Hermes]: {reply}")
                    speak(reply)
                    # MUST come after sd.wait(): the assistant's own voice is
                    # sitting in the mic buffer and would ghost-trigger the wake.
                    drain_mic()
                continue

    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
