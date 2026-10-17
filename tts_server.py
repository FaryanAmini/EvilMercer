"""Local Qwen3-TTS server for EvilMercer's speak().

Runs in its OWN env (torch/CUDA needs Python 3.12; Hermes' pm interpreter is
3.14 and stays untouched):

    cd /home/faryan/projects/EvilMercer
    uv venv --python 3.12 .venv-tts
    uv pip install --python .venv-tts qwen-tts
    # optional but recommended (big speed win; long one-time CUDA compile):
    uv pip install --python .venv-tts ninja packaging wheel
    MAX_JOBS=4 uv pip install --python .venv-tts flash-attn --no-build-isolation
    .venv-tts/bin/python tts_server.py        # first run downloads model weights

Serves POST http://127.0.0.1:8055/tts  {"text": "..."}  -> WAV bytes.
"""
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
# CustomVoice timbres are the stable ones; a heavy free-form INSTRUCT competes
# with the speaker embedding and adds drift -- keep it empty unless experimenting.
SPEAKER = os.getenv("TTS_SPEAKER", "Dylan")
PORT = 8055
INSTRUCT = os.getenv("TTS_INSTRUCT", "")

# Voice clone: drop voice/ref.wav + voice/ref.txt (your recording + its exact
# transcript) and the server clones your voice with the Base model instead of
# using the preset speaker. Remove the files or set TTS_NO_CLONE=1 to revert.
REF_AUDIO = os.getenv("TTS_REF_AUDIO", "voice/ref.wav")
REF_TEXT = os.getenv("TTS_REF_TEXT", "voice/ref.txt")

GEN_LOCK = threading.Lock()  # model.generate is not thread-safe


def load_model(model_id):
    """flash-attn is optional; fall back to sdpa if it isn't installed."""
    last = None
    for attn in ("flash_attention_2", "sdpa"):
        try:
            return Qwen3TTSModel.from_pretrained(
                model_id,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation=attn,
            )
        except Exception as e:  # ImportError/ValueError for missing flash_attn
            print(f"[tts_server] attn_implementation={attn} failed: {e}")
            last = e
    raise RuntimeError(f"could not load {model_id}: {last}")


def build_backend():
    """Clone mode if reference files exist, else preset CustomVoice mode."""
    if not os.getenv("TTS_NO_CLONE") and os.path.isfile(REF_AUDIO) and os.path.isfile(REF_TEXT):
        ref_text = open(REF_TEXT).read().strip()
        print(f"[tts_server] cloning voice from {REF_AUDIO} (Base model)...")
        model = load_model(BASE_MODEL_ID)
        prompt = model.create_voice_clone_prompt(
            ref_audio=REF_AUDIO, ref_text=ref_text, x_vector_only_mode=False
        )
        return "clone", model, prompt
    print(f"[tts_server] loading preset speaker {SPEAKER} (CustomVoice)...")
    return "custom", load_model(MODEL_ID), None


MODE, MODEL, CLONE_PROMPT = build_backend()
# Warm up CUDA kernels / Triton JIT once at boot, so the first real reply
# doesn't pay the multi-second compile cost of a cold generation.
if MODE == "clone":
    MODEL.generate_voice_clone(text="Ready.", language="English",
                               voice_clone_prompt=CLONE_PROMPT)
else:
    MODEL.generate_custom_voice(text="Ready.", language="English", speaker=SPEAKER)
print(f"[tts_server] model ready ({MODE}).")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/tts":
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with GEN_LOCK:
                # Low-temp sampling with a text-derived seed: greedy (the old
                # do_sample=False) makes small TTS models stop early (EOS)
                # mid-sentence; sampling fixes truncation, and the seed keeps
                # repeated/similar lines consistent across calls.
                import zlib
                seed = zlib.crc32(body["text"].encode()) & 0x7FFFFFFF
                generator = torch.Generator(device="cuda").manual_seed(seed)
                gen_kwargs = dict(
                    language="English",  # fixed per request; do not auto-detect
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=50,
                    generator=generator,
                )
                if MODE == "clone":
                    wavs, sr = MODEL.generate_voice_clone(
                        text=body["text"],
                        voice_clone_prompt=CLONE_PROMPT,
                        **gen_kwargs,
                    )
                else:
                    wavs, sr = MODEL.generate_custom_voice(
                        text=body["text"],
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
