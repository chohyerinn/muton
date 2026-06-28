# MUTON Evaluation Pipeline

This document describes the runtime evaluation pipeline used for Graduation Project 2.

## What It Measures

The evaluation script measures four parts of the current MUTON system:

- STT quality: CER, WER, confidence, and audio chunk latency
- summary quality: generated summary per model/mode
- ablation: `text`, `text_face`, `text_audio`, and `full`
- LoRA effect: `base` vs `lora`

The script writes CSV/JSON outputs that can be copied into the report.

## Enable Evaluation Endpoint

The ablation and base-vs-LoRA tests use a separate endpoint:

```text
POST /eval/generate_summary
```

This endpoint is disabled by default so it is not accidentally exposed during a public demo. Enable it only while running evaluation:

```bash
export MUTON_ENABLE_EVAL_ENDPOINTS=true
```

## Start The Server

```bash
cd ~/MUTON_cpy
git checkout server_main
git pull --rebase origin server_main

source ~/miniconda3/etc/profile.d/conda.sh
conda activate muton

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export MUTON_QWEN_STT_BACKEND=openai
export MUTON_QWEN_ADAPTER=/home/jaesang02/MUTON_cpy/out/qwen_omni_lora/ko_stage
export MUTON_ENABLE_EVAL_ENDPOINTS=true

CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen_server.py
```

## Prepare Manifest

Copy the example manifest and replace paths with real samples:

```bash
cp examples/evaluation_manifest.example.json out/eval_manifest.json
```

Example sample:

```json
{
  "id": "quiet_meeting_001",
  "image_path": "/path/to/frame.jpg",
  "audio_path": "/path/to/utterance.wav",
  "audio_format": "wav",
  "reference_text": "Korean reference transcript here.",
  "reference_summary": "Korean reference summary here."
}
```

Audio requirements:

- WAV: `16kHz`, mono or stereo, 16-bit PCM
- PCM: raw `16kHz`, mono, `int16`

## Prepare Manifest From MELD

If you only have MELD Raw mp4 files, generate evaluation samples from the MELD CSV:

```bash
python scripts/prepare_meld_eval_manifest.py \
  --csv data/MELD/MELD.Raw/train_sent_emo.csv \
  --videos_root data/MELD/MELD.Raw/train_splits \
  --output_dir out/eval_meld_samples \
  --manifest out/eval_manifest_meld.json \
  --max_samples 20 \
  --emotion_balanced \
  --translate \
  --make_reference_summary
```

Depending on the local MELD layout, `--videos_root` may be one of:

```text
data/MELD/MELD.Raw/train_splits
data/MELD/MELD.Raw/dev_splits_complete
data/MELD/MELD.Raw/output_repeated_splits_test
data/MELD/MELD.Raw
```

The script creates:

```text
out/eval_meld_samples/frames/*.jpg
out/eval_meld_samples/audio/*.wav
out/eval_manifest_meld.json
```

If `--translate` is omitted, `reference_text` keeps the English MELD utterance. For the MUTON report, Korean translated text is recommended.

## Run Evaluation

```bash
python scripts/evaluate_runtime_pipeline.py \
  --base_url http://127.0.0.1:5000 \
  --manifest out/eval_manifest.json \
  --output_dir out/eval_runtime
```

For Cloudflare:

```bash
python scripts/evaluate_runtime_pipeline.py \
  --base_url https://xxxxx.trycloudflare.com \
  --manifest out/eval_manifest.json \
  --output_dir out/eval_runtime_cloudflare
```

## Outputs

```text
out/eval_runtime/
  stt_results.csv
  summary_results.csv
  summary_human_eval_template.csv
  results.json
  report.md
```

`stt_results.csv` contains:

- reference text
- predicted text
- CER
- WER
- STT confidence
- chunk latency
- total STT latency

`summary_results.csv` contains:

- sample id
- mode: `text`, `text_face`, `text_audio`, `full`
- variant: `base` or `lora`
- generated summary
- ROUGE-L F1 when reference summary exists
- summary latency

`summary_human_eval_template.csv` adds blank 1-5 human scoring columns:

- emotion reflection
- intent reflection
- fluency
- faithfulness

## Recommended Report Tables

Use the outputs to make these report tables:

1. STT comparison: `whisper-1` vs local Korean Whisper
2. LoRA comparison: Qwen2.5-Omni base vs Qwen2.5-Omni + `ko_stage` LoRA
3. Modality ablation: text-only vs text+face vs text+audio vs full multimodal
4. Runtime latency: audio chunk, STT finalization, summary generation
