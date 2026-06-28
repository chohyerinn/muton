# MUTON Wiki

MUTON is a real-time omnimodal dialogue assistance system for hearing-impaired users, especially users who rely on oral communication and context reading in daily conversation. The system combines speech, facial cues, and transcript context to provide subtitles and short Korean summaries that include conversational tone and intent.

<img src="../docs/assets/muton_poster.jpg" alt="MUTON project poster" width="850" />

## Recommended Runtime

The current Graduation Project 2 runtime is:

- STT: `OpenAI whisper-1`
- multimodal summary: `Qwen2.5-Omni + ko_stage LoRA`
- backend: `FastAPI`
- mobile endpoint discovery: `backend_url.json` on the `server_main` branch
- Android client: [Ai-pre/MUTON-Android](https://github.com/Ai-pre/MUTON-Android)

This structure separates transcription from multimodal reasoning. Whisper handles subtitle quality, while Qwen2.5-Omni handles image/audio/text-based summary generation.

## Project Evolution

P-project focused on building a full multimodal pipeline directly: face, audio, and text encoders were connected to a custom fusion Transformer. That structure was important because it proved that the Android app, backend server, and multimodal inference flow could work together.

Graduation Project 2 keeps the same service goal but changes the modeling strategy. Instead of compressing every modality into external encoder features and then generating summaries through a limited template-like path, the current system uses a pretrained multimodal generation model. This makes the runtime closer to a practical service pipeline and allows raw image, audio, and transcript context to be used more naturally.

## Main Features

- real-time camera frame ingestion from Android
- utterance-level speech buffering with VAD
- Korean subtitle generation through STT
- face-based visual emotion output for the app UI
- multimodal Korean summary generation through Qwen2.5-Omni
- server-side conversation record summary to avoid exposing API keys in the Android app
- Cloudflare Tunnel based mobile access

## Final Project Results

- LoRA adaptation improves the output style from long chatbot-like responses to shorter Korean observational captions.
- The best automatic summary score is `ROUGE-L F1 = 0.1616` for the Text + Face LoRA setting.
- The full Text + Face + Audio LoRA setting reaches `ROUGE-L F1 = 0.1405`.
- Human evaluation score improves from `10.73 / 20` to `16.60 / 20`.
- STT server processing averages `1.4071s` over 300 samples.
- Mobile end-to-end latency averages `5.6s` over 10 live Android utterances.

## Current Limitations

- speaker diarization is not yet implemented for conversations with three or more participants
- STT and utterance segmentation still require tuning for noisy environments
- end-to-end latency requires further optimization
- target-user evaluation with hearing-impaired participants remains future work

## Important Entry Points

- `scripts/run_qwen_server.py`: starts the current FastAPI server
- `src/server_qwen.py`: current Qwen runtime, API endpoints, and summary flow
- `src/encoders.py`: face, audio, STT, and feature extraction utilities
- `scripts/update_backend_url.py`: updates the Android backend discovery file
- `backend_url.json`: public runtime URL consumed by the Android app

## Reading Order

1. [Architecture](Architecture)
2. [Installation](Installation)
3. [Datasets](Datasets)
4. [Training](Training)
5. [API](API)
6. [Examples](Examples)
7. [Evaluation](Evaluation)
