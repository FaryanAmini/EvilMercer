"""Local Qwen3-TTS server for EvilMercer's speak().

Runs in its OWN env (torch/CUDA needs Python 3.12; Hermes' pm interpreter is
3.14 and stays untouched):

    cd /home/faryan/projects/EvilMercer
    uv venv --python 3.12 .venv-tts
    .venv-tts/bin/pip install qwen-tts
    .venv-tts/bin/python tts_server.py        # first run downloads model weights

Serves POST http://127.0.0.1:8055/tts  {"text": "..."}  -> WAV bytes.
"""
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
# CustomVoice timbres are the stable ones; a heavy free-form INSTRUCT competes
# with the speaker embedding and adds drift -- keep it empty unless experimenting.
SPEAKER = os.getenv("TTS_SPEAKER", "Dylan")
PORT = 8055
INSTRUCT = os.getenv("TTS_INSTRUCT", "")

GEN_LOCK = threading.Lock()  # model.generate is not thread-safe


def load_model():
    """flash-attn is optional; fall back to sdpa if it isn't installed."""
    last = None
    for attn in ("flash_attention_2", "sdpa"):
        try:
            return Qwen3TTSModel.from_pretrained(
                MODEL_ID,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation=attn,
            )
        except Exception as e:  # ImportError/ValueError for missing flash_attn
            print(f"[tts_server] attn_implementation={attn} failed: {e}")
            last = e
    raise RuntimeError(f"could not load {MODEL_ID}: {last}")


print(f"[tts_server] loading {MODEL_ID} (speaker={SPEAKER})...")
MODEL = load_model()
print("[tts_server] model ready.")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/tts":
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with GEN_LOCK:
                gen_kwargs = {"do_sample": False} if not body.get("sample") else {}
                # 0.6B-CustomVoice has no instruction control (1.7B does) --
                # only pass the speaker's preset timbre.
                if INSTRUCT or body.get("instruct"):
                    print("[tts_server] note: 0.6B ignores 'instruct'; only 1.7B supports it")
                wavs, sr = MODEL.generate_custom_voice(
                    text=body["text"],
                    language="English",  # fixed
                    speaker=SPEAKER,
                    **gen_kwargs,
                )
            buf = io.BytesIO()
            sf.write(buf, wavs[0], sr, format="WAV")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, f"tts failed: {e}")

    def log_message(self, *args):  # keep stdout clean for generation logs
        pass


if __name__ == "__main__":
    print(f"[tts_server] listening on http://127.0.0.1:{PORT}/tts")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
