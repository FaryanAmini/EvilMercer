#!/bin/sh
# EvilMercer launcher. App deps live in ./app_libs (pip --target) so they never
# touch Hermes' pm/uv-managed environment -- installing into that venv conflicts
# with uv.lock's pins (and pm wipes/re-syncs it whenever it likes).
set -e
cd "$(dirname "$0")"
. ./hermes-agent/activate
if ! PYTHONPATH="$PWD/app_libs${PYTHONPATH:+:$PYTHONPATH}" \
     python3 -c "import numpy, pyaudio, openwakeword, moonshine_voice, dashscope" 2>/dev/null; then
    echo "[run.sh] app deps missing -- installing into ./app_libs"
    python3 -m pip install -q --target app_libs -r requirements.txt
fi
PYTHONPATH="$PWD/app_libs${PYTHONPATH:+:$PYTHONPATH}" exec python3 main.py
