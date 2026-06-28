# Architecture

## System Evolution

MUTON has two major architectural stages.

## P-project Pipeline

<img width="1187" height="447" alt="MUTON P-project pipeline" src="https://github.com/user-attachments/assets/b0e53b3b-c230-44fe-8d02-fdef0f2bf222" />

The P-project pipeline used separate face, audio, and text encoders followed by a directly designed multimodal fusion Transformer.

- Face path: face detection, alignment, ViT-based facial emotion features
- Audio path: VAD, STT, WavLM-based speech features
- Text path: Korean sentence embedding through a language encoder
- Fusion path: encoder-only Transformer-based multimodal representation
- Summary path: structured information was passed to a generation step

This version proved the feasibility of the full system, but summary quality was limited by feature compression, limited training data, and a constrained generation flow.

## Graduation Project 2 Pipeline

<img width="1237" height="395" alt="MUTON Graduation Project 2 pipeline" src="https://github.com/user-attachments/assets/cc9feac7-ec11-4e39-b25d-b5e6a9ac3db5" />

Graduation Project 2 keeps the Android streaming and FastAPI server structure, but the summary engine changed. The current runtime uses `whisper-1` for STT and `Qwen2.5-Omni + ko_stage LoRA` for multimodal summary generation.

The key change is that raw multimodal context is now preserved longer. Instead of relying only on externally extracted encoder vectors, the server commits an utterance snapshot containing:

- finalized transcript
- utterance waveform
- latest face image near the finalized utterance
- STT confidence

Qwen2.5-Omni then generates the final Korean summary from this committed snapshot.

## Current Runtime Flow

1. Android streams camera frames to `/process_video_chunk`.
2. Android streams PCM audio chunks to `/process_audio_chunk`.
3. The server uses VAD and buffering to detect utterance boundaries.
4. STT produces a finalized Korean transcript.
5. The server commits a synchronized utterance snapshot.
6. Android calls `/get_fusion_analysis`.
7. Qwen2.5-Omni generates the multimodal summary.
8. Android displays subtitle, visual emotion, and summary output.

## Why STT And Summary Are Split

STT and multimodal reasoning have different reliability requirements. The current system keeps them separate because:

- `whisper-1` is more reliable for Korean subtitle transcription in the live demo.
- Qwen2.5-Omni is stronger as a multimodal reasoning and summary model.
- Separating STT from summary makes latency and failure points easier to control.
- The STT backend can be replaced without redesigning the summary model.
- The Android app can keep the same UI even when backend model experiments change.

## Confidence Handling

The server suppresses unreliable outputs before they become misleading summaries.

- Very low-confidence transcripts are filtered at the STT stage.
- Low committed STT confidence returns `Low Confidence` from `/get_fusion_analysis`.
- Repeated-token and common hallucination patterns are filtered in the audio encoder path.

This is especially important in noisy environments, where a wrong transcript can produce a confident but incorrect summary.
