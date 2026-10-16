import threading
import time
import pyaudio
import numpy as np
from openwakeword.model import Model
from moonshine_voice import MicTranscriber, ModelArch, LineTextChanged, LineCompleted

import openwakeword

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
transcriber.on_text(lambda text: print(f"[You said]: {text}"))
transcriber.load()

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
        #wakeModel.reset() # commented out as it was causing wake word looping

    return session["final_text"]

def main():
    try:
        while True:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            # Convert raw bytes to a 16-bit numpy array
            audio_frame = np.frombuffer(raw_data, dtype=np.int16)
            # Feed the frame into the model
            # openWakeWord handles internal buffering automatically
            prediction = wakeModel.predict(audio_frame)
            # Adjust threshold (e.g., 0.5) to fine-tune sensitivity
            if prediction["alexa_v0.1"] > 0.5:
                print("\n[!] Wake Word Detected! Activation triggered.")

                # Hand off to Moonshine for transcription of the user's command
                run_assistant_session()

                # Reset wake word model state so it doesn't retrigger immediately
                wakeModel.reset()
                continue

    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
