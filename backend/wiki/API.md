# API

The current backend is implemented in `src/server_qwen.py`.

Base URL examples:

```text
http://127.0.0.1:5000
https://xxxxx.trycloudflare.com
```

FastAPI interactive docs:

```text
http://127.0.0.1:5000/docs
```

## `GET /health`

Checks whether the backend is alive.

Response:

```json
{
  "status": "ok",
  "backend": "qwen_omni"
}
```

## `POST /process_video_chunk`

Receives one JPEG frame, extracts face/emotion context, and caches the latest face image for the next summary step.

Request:

- Content-Type: `multipart/form-data`
- `frame`: JPEG image file

Response:

```json
{
  "status": "ok",
  "image_source": "face_crop",
  "emotion": "Happy"
}
```

Notes:

- `emotion` is the visual emotion label shown by the mobile UI.
- The cached face image is used when an utterance snapshot is committed.

## `POST /process_audio_chunk`

Receives raw PCM audio chunks. The server buffers audio until an utterance boundary is detected, then runs STT.

Request:

- Content-Type: `multipart/form-data`
- `audio`: raw PCM bytes, `16kHz`, mono, `int16`

Response while the utterance is still open:

```json
{
  "text": "",
  "stt_confidence": 0.0,
  "prosody": [],
  "content": [],
  "speaker": [],
  "fusion_emotion": "",
  "summary": ""
}
```

Response when an utterance is finalized:

```json
{
  "text": "오늘 회의는 조금 급하게 진행되는 것 같아요.",
  "stt_confidence": 0.82,
  "prosody": [],
  "content": [],
  "speaker": [],
  "fusion_emotion": "",
  "summary": ""
}
```

Notes:

- `text` remains empty until the server considers the utterance complete.
- `stt_confidence` is used by the summary stage to avoid unreliable outputs.
- `prosody`, `content`, and `speaker` remain in the response for Android compatibility with earlier fusion experiments.

## `POST /get_fusion_analysis`

Generates a Korean multimodal summary from the committed utterance snapshot.

Committed snapshot:

- finalized transcript
- utterance waveform
- latest face image at utterance-final time
- STT confidence

Request:

- Content-Type: `multipart/form-data`
- `text`: transcript string
- `prosody`: JSON string, usually `"[]"`
- `content`: JSON string, usually `"[]"`
- `speaker`: JSON string, usually `"[]"`

Successful response:

```json
{
  "fusion_emotion": "",
  "fusion_confidence": 0.81,
  "arousal": 0.0,
  "valence": 0.0,
  "summary": "상대방이 급한 분위기에서 회의 진행 상황을 설명하고 있습니다.",
  "cls_attn": []
}
```

Low-confidence response:

```json
{
  "fusion_emotion": "Low Confidence",
  "fusion_confidence": 0.31,
  "arousal": 0.0,
  "valence": 0.0,
  "summary": "",
  "cls_attn": []
}
```

No visual input response:

```json
{
  "fusion_emotion": "No Visual Input",
  "summary": ""
}
```

## `POST /summarize_conversation_record`

Summarizes saved conversation text on the backend. This endpoint exists so the Android app does not need to store an OpenAI API key.

Request:

- Content-Type: `application/json`

```json
{
  "conversation_text": "speaker: 오늘 회의는 조금 급하게 진행되는 것 같아요.\nlistener: 네, 핵심만 먼저 정리해 주세요."
}
```

Response:

```json
{
  "title": "회의 진행 상황을 빠르게 정리하는 대화"
}
```

Failure response:

```json
{
  "title": "",
  "error": "summary_failed"
}
```

## `POST /eval/generate_summary`

Generates a summary from explicitly supplied evaluation inputs. This endpoint is intended for benchmarking and ablation studies, not for the Android runtime.

It is disabled by default. Enable it before starting the server:

```bash
export MUTON_ENABLE_EVAL_ENDPOINTS=true
```

Request:

- Content-Type: `multipart/form-data`
- `text`: transcript string
- `mode`: `text`, `text_face`, `text_audio`, or `full`
- `use_adapter`: `true` for LoRA, `false` for base model
- `frame`: JPEG image file when the mode uses face input
- `audio`: raw PCM or WAV bytes when the mode uses audio input
- `audio_format`: `pcm` or `wav`

Response:

```json
{
  "summary": "상대방이 급한 분위기에서 회의 진행 상황을 설명하고 있습니다.",
  "mode": "full",
  "use_adapter": true,
  "face_used": true,
  "audio_used": true,
  "latency_sec": 4.21,
  "model": "Qwen/Qwen2.5-Omni-7B",
  "adapter": "/path/to/out/qwen_omni_lora/ko_stage"
}
```

## Recommended Runtime Configuration

```bash
export MUTON_QWEN_STT_BACKEND=openai
export MUTON_RECORD_SUMMARY_MODEL=gpt-4o
```

The current recommended path uses `whisper-1` for STT and Qwen2.5-Omni for multimodal summary generation.
