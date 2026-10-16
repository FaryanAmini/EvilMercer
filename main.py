import time
import pyaudio
import numpy as np
from openwakeword.model import Model
from moonshine_voice import MicTranscriber, ModelArch

wakeModel = Model(wakeword_models=["alexa"])

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
transcriber = MicTranscriber(model_arch=ModelArch.TINY)

def run_assistant_session():
    """Capture and transcribe speech after the wake word is detected."""
    print("[*] Listening... (press Ctrl+C to end the session)")

    def on_transcription(text: str):
        print(f"[You said]: {text}")
        # Insert smart assistant / LLM response logic here

    try:
        # Blocks until interrupted or stopped; streams mic audio to Moonshine
        transcriber.start(on_transcription)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            transcriber.stop()
        except Exception:
            pass

def main():
    try:
        while True:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            # Convert raw bytes to a 16-bit numpy array
            audio_frame = np.frombuffer(raw_data, dtype=np.int16)
            # Feed the frame into the model
            # openWakeWord handles internal buffering automatically
            prediction = wakeModel.predict(audio_frame)
            # Check prediction probability for the chosen wake word
            # Adjust threshold (e.g., 0.5) to fine-tune sensitivity
            if prediction["alexa"] > 0.5:
                print("\n[!] Wake Word Detected! Activation triggered.")

                # Hand off to Moonshine for transcription of the user's command
                run_assistant_session()

                # Reset wake word model state so it doesn't retrigger immediately
                wakeModel.reset()

    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
