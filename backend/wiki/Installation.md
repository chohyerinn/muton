# Installation

## Requirements

- Python `3.10+`
- CUDA-capable GPU for Qwen2.5-Omni inference
- OpenAI API key for the recommended `whisper-1` STT path
- `cloudflared` for public Android access
- Git branch: `server_main`

## Main Libraries

The backend uses:

- `fastapi`
- `uvicorn`
- `torch`
- `torchaudio`
- `transformers`
- `accelerate`
- `peft`
- `openai`
- `opencv-python`
- `mediapipe`
- `librosa`
- `soundfile`
- `Pillow`

Dependency files:

- `requirements.txt`
- `requirements-qwen-omni.txt`

## Environment Setup

```bash
cd ~/MUTON_cpy
git checkout server_main
git pull --rebase origin server_main

pip install -r requirements.txt
pip install -r requirements-qwen-omni.txt
```

## Runtime Variables

Recommended live-demo configuration:

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export MUTON_QWEN_STT_BACKEND=openai
export MUTON_QWEN_ADAPTER=/path/to/out/qwen_omni_lora/ko_stage
```

Optional variables:

```bash
export MUTON_RECORD_SUMMARY_MODEL=gpt-4o
export MUTON_STT_SUMMARY_MIN_CONFIDENCE=0.55
export MUTON_QWEN_STT_BACKEND=local
```

Use `MUTON_QWEN_STT_BACKEND=openai` for the current recommended runtime. The local Korean Whisper backend remains available as a fallback or comparison path.

## Start The Server

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen_server.py
```

Health check:

```bash
curl http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "status": "ok",
  "backend": "qwen_omni"
}
```

## Cloudflare Tunnel

Expose the local server:

```bash
cloudflared tunnel --url http://127.0.0.1:5000
```

When the tunnel URL changes, update `backend_url.json`:

```bash
python scripts/update_backend_url.py https://xxxxx.trycloudflare.com
git add backend_url.json
git commit -m "Update backend URL"
git push origin server_main
```

Android reads:

```text
https://raw.githubusercontent.com/Ai-pre/MUTON/server_main/backend_url.json
```

## Common Runtime Checks

- If the app only reaches `/health`, the backend URL is correct but audio/video requests may not be active yet.
- If the server logs `address already in use`, another process is already bound to port `5000`.
- If the local Whisper model loads unexpectedly, check that `MUTON_QWEN_STT_BACKEND=openai` is exported in the same shell that starts the server.
- If API authentication fails, verify `OPENAI_API_KEY` on the server side. The Android app should not contain this key.
