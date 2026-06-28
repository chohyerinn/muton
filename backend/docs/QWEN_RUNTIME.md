# Qwen Runtime Runbook

## Recommended Deployment Split

- STT: `OpenAI whisper-1`
- summary/reasoning: `Qwen2.5-Omni + ko_stage LoRA`
- mobile transport: FastAPI + Cloudflare tunnel + `backend_url.json`

This split exists because the local Korean Whisper fallback is cheaper but degrades badly in noisy environments, while Qwen is strongest when used for multimodal reasoning rather than raw subtitle transcription.

## Server Start

```bash
cd ~/MUTON_cpy
git fetch origin
git checkout codex/server_main
git reset --hard origin/codex/server_main

source ~/miniconda3/etc/profile.d/conda.sh
conda activate muton

export OPENAI_API_KEY=YOUR_KEY
export MUTON_QWEN_ADAPTER=/home/jaesang02/MUTON_cpy/out/qwen_omni_lora/ko_stage
export MUTON_QWEN_STT_BACKEND=openai

CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen_server.py
```

## STT Backend Modes

### `MUTON_QWEN_STT_BACKEND=openai`

Use this when subtitle quality matters most.

Benefits:

- restores the earlier `whisper-1` API path
- uses OpenAI transcript metadata such as `avg_logprob` and `no_speech_prob`
- usually behaves best in noisy real-world audio

### `MUTON_QWEN_STT_BACKEND=whisper`

Use this when API cost matters more than quality.

Benefits:

- fully local
- Korean-tuned fallback model

Tradeoff:

- more repeated garbage transcripts in noisy scenes

### `MUTON_QWEN_STT_BACKEND=qwen`

Keep this for experiments only. It is not the recommended mobile subtitle path.

## Sync Between STT And Summary

The current server snapshots one utterance as a bundle:

1. buffered audio reaches utterance-final state through VAD
2. transcript is finalized
3. that utterance waveform is frozen
4. the latest face image at that moment is frozen
5. `/get_fusion_analysis` uses only that committed snapshot

This means the summary is no longer generated from a newer transcript and an older audio cache. Audio/text sync is now tied to utterance finalization; face remains the nearest-frame approximation.

## Confidence Flow

`src/encoders.py` now computes transcript confidence.

- if transcript confidence is too low, the subtitle is dropped
- if committed STT confidence is below `MUTON_STT_SUMMARY_MIN_CONFIDENCE`, `/get_fusion_analysis` returns `Low Confidence`
- the Android app can then avoid showing a misleading summary

Useful knobs:

```bash
export MUTON_STT_MIN_TRANSCRIPT_CONFIDENCE=0.45
export MUTON_STT_SUMMARY_MIN_CONFIDENCE=0.55
```

Raise them to be more conservative. Lower them if too many utterances disappear.

## Remote URL Update

The Android app should read:

```text
https://raw.githubusercontent.com/Ai-pre/MUTON/refs/heads/server/backend_url.json
```

To update the active tunnel URL:

```bash
cd ~/MUTON_server
python scripts/update_backend_url.py https://xxxxx.trycloudflare.com
git add backend_url.json
git commit -m "Update backend URL"
git push origin HEAD:server
```

Verify with cache busting:

```bash
curl "https://raw.githubusercontent.com/Ai-pre/MUTON/refs/heads/server/backend_url.json?t=$(date +%s)"
```

## Troubleshooting

### `Both max_new_tokens and max_length seem to have been set`

This is a local Whisper generation-config issue, not a sign that the spoken utterance is too short.

### `Incoming request ended abruptly: context canceled`

This usually means the mobile client canceled the request before the origin finished responding. In practice it is often caused by overlapping chunk requests or STT latency spikes rather than a Cloudflare fault.

### Repeated garbage such as `gosokdorogosokdoro` or `nagatnagatnagat`

These are handled through:

- utterance-level VAD gating
- repeated-span filtering
- transcript confidence filtering

If they still leak through, prefer `MUTON_QWEN_STT_BACKEND=openai`.
