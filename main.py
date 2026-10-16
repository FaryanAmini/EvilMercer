import time
import pyaudio
import numpy as np
from openwakeword.model import Model
from moonshine_voice import MicTranscriber, ModelArch

wakeModel = Model(wakeword_models=["alexa"])

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280  # Determines the frame step sizing

audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

def main():
    try:
        while True:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            # Convert raw bytes to a 16-bit numpy array
            audio_frame = np.frombuffer(raw_data, dtype=np.int16)
            # Feed the frame into the model
            # openWakeWord handles internal buffering automatically
            prediction = model.predict(audio_frame)
            # Check prediction probability for the chosen wake word
            # Adjust threshold (e.g., 0.5) to fine-tune sensitivity
            if prediction["alexa"] > 0.5:
                print("\n[!] Wake Word Detected! Activation triggered.")

                # Insert  voice assistant logic

    except KeyboardInterrupt:
        print("\nstoping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ = "__main__":
    main()
