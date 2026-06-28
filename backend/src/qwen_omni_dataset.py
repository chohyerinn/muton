from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image

from muton.config import env_path
from muton.encoders import FaceEncoder
from src.build_rich_dataset import (
    crop_face_bgr,
    extract_audio_wav_with_ffmpeg,
    extract_meld_frames,
    get_summary_text,
    load_meld_time_ranges,
    parse_meld_video_path,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a multimodal emotion and situation analyst. "
    "Use the face image, audio, and script together. "
    "Answer in natural Korean with one concise observational sentence."
)

DEFAULT_INSTRUCTION = (
    "얼굴 이미지, 음성, 대사를 함께 참고해서 화자의 감정, 태도, 상황을 한국어 한 문장으로 설명해라. "
    "보이는 표정과 들리는 말투를 바탕으로 쓰되 과장하지 마라."
)


def ensure_parent(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    out_path = ensure_parent(path)
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_user_text(script: str, instruction: str, emotion: str = "") -> str:
    lines = [
        instruction.strip(),
        "",
        f"대사: {script.strip()}",
    ]
    if emotion:
        lines.append(f"참고 감정 라벨: {emotion.strip()}")
    return "\n".join(lines).strip()


def build_messages(
    image_paths: List[str],
    audio_path: str,
    script: str,
    target_text: str,
    instruction: str,
    system_prompt: str,
    emotion: str = "",
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, str]] = []
    for image_path in image_paths:
        user_content.append({"type": "image", "path": image_path})
    user_content.append({"type": "audio", "path": audio_path})
    user_content.append({"type": "text", "text": build_user_text(script, instruction, emotion=emotion)})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt.strip()}],
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_text.strip()}],
        },
    ]


def load_audio_array(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    waveform, sample_rate = sf.read(audio_path)
    if getattr(waveform, "ndim", 1) > 1:
        waveform = waveform.mean(axis=1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if sample_rate != target_sr:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
    return waveform.astype(np.float32, copy=False)


def materialize_messages(messages: List[Dict[str, Any]], audio_sampling_rate: int = 16000) -> List[Dict[str, Any]]:
    materialized: List[Dict[str, Any]] = []
    for message in messages:
        content_items = message.get("content")
        if not isinstance(content_items, list):
            materialized.append(message)
            continue

        new_items: List[Dict[str, Any]] = []
        for item in content_items:
            item_type = item.get("type", "")
            if item_type == "image":
                image_path = item.get("image") or item.get("path")
                if image_path:
                    image = Image.open(str(image_path)).convert("RGB")
                    new_items.append({"type": "image", "image": image})
                    continue
            if item_type == "audio":
                audio_path = item.get("audio") or item.get("path")
                if audio_path:
                    audio = load_audio_array(str(audio_path), target_sr=audio_sampling_rate)
                    new_items.append({"type": "audio", "audio": audio})
                    continue
            new_items.append(item)

        new_message = dict(message)
        new_message["content"] = new_items
        materialized.append(new_message)
    return materialized


def save_face_crops(
    face_encoder: FaceEncoder,
    frames_bgr: List[Any],
    image_root: Path,
    sample_id: str,
) -> List[str]:
    image_root.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for frame_idx, frame_bgr in enumerate(frames_bgr):
        crop_bgr = crop_face_bgr(face_encoder, frame_bgr)
        if crop_bgr is None:
            continue
        image_path = image_root / f"{sample_id}_f{frame_idx}.jpg"
        cv2.imwrite(str(image_path), crop_bgr)
        saved_paths.append(str(image_path))

    return saved_paths


def export_meld_manifest(
    input_pt: str,
    videos_root: str,
    meld_csv: str,
    out_jsonl: str,
    media_root: str,
    num_frames: int,
    instruction: str,
    system_prompt: str,
    include_emotion_label: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    data = torch.load(input_pt, map_location="cpu", weights_only=False)
    time_ranges = load_meld_time_ranges(meld_csv)
    videos_root_path = Path(videos_root)
    media_root_path = Path(media_root)
    image_root = media_root_path / "images"
    audio_root = media_root_path / "audio"

    face_encoder = FaceEncoder()
    records: List[Dict[str, Any]] = []
    attempted = 0

    for sample in data:
        if limit > 0 and attempted >= limit:
            break
        attempted += 1

        sample_id = str(sample.get("id", "")).strip()
        target_text = str(sample.get("target_text", "")).strip()
        if not sample_id or not target_text:
            continue

        video_path = parse_meld_video_path(videos_root_path, sample_id)
        if not video_path.exists():
            print(f"[skip] missing video: {sample_id} -> {video_path}")
            continue

        start_t, end_t = time_ranges.get(sample_id, (0.0, 0.0))
        frames_bgr = extract_meld_frames(video_path, start_t, end_t, num_frames=num_frames)
        if not frames_bgr:
            print(f"[skip] frame read fail: {sample_id}")
            continue

        image_paths = save_face_crops(face_encoder, frames_bgr, image_root=image_root, sample_id=sample_id)
        if not image_paths:
            print(f"[skip] face crop fail: {sample_id}")
            continue

        audio_path = audio_root / f"{sample_id}.wav"
        if not audio_path.exists():
            try:
                extract_audio_wav_with_ffmpeg(video_path, audio_path, sr=16000)
            except Exception as exc:
                print(f"[skip] audio extract fail: {sample_id} / {exc}")
                continue

        emotion = str(sample.get("emotion", "")) if include_emotion_label else ""
        record = {
            "id": sample_id,
            "image_paths": image_paths,
            "audio_path": str(audio_path),
            "script": str(sample.get("script", "")),
            "target_text": target_text,
            "emotion": str(sample.get("emotion", "")),
            "messages": build_messages(
                image_paths=image_paths,
                audio_path=str(audio_path),
                script=str(sample.get("script", "")),
                target_text=target_text,
                instruction=instruction,
                system_prompt=system_prompt,
                emotion=emotion,
            ),
        }
        records.append(record)

        if len(records) % 25 == 0:
            print(f"[ok] exported {len(records)} samples (attempted={attempted})")

    write_jsonl(out_jsonl, records)
    print(f"saved: {out_jsonl}")
    print(f"generated: {len(records)}")
    print(f"attempted: {attempted}")
    return records


def export_ko_manifest(
    json_path: str,
    face_root: str,
    audio_root: str,
    out_jsonl: str,
    instruction: str,
    system_prompt: str,
    include_emotion_label: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    face_root_path = Path(face_root)
    audio_root_path = Path(audio_root)

    records: List[Dict[str, Any]] = []
    attempted = 0
    for clip_key, items in data.items():
        if not str(clip_key).startswith("clip_"):
            continue

        person_id = str(clip_key).replace("clip_", "").strip()
        face_dir = face_root_path / person_id
        audio_dir = audio_root_path / person_id
        if not face_dir.is_dir() or not audio_dir.is_dir():
            continue

        for item in items:
            if limit > 0 and attempted >= limit:
                write_jsonl(out_jsonl, records)
                print(f"saved: {out_jsonl}")
                print(f"generated: {len(records)}")
                print(f"attempted: {attempted}")
                return records

            sample_id = str(item.get("name", "")).strip()
            target_text = get_summary_text(item)
            if not sample_id or not target_text:
                continue
            attempted += 1

            face_path = face_dir / f"{sample_id}.jpg"
            if not face_path.exists():
                face_path = face_dir / f"{sample_id}.jpeg"
            audio_path = audio_dir / f"{sample_id}.wav"
            if not face_path.exists() or not audio_path.exists():
                print(f"[skip] missing media: {sample_id}")
                continue

            emotion = str(item.get("emotion", "")) if include_emotion_label else ""
            record = {
                "id": sample_id,
                "person_id": person_id,
                "image_paths": [str(face_path)],
                "audio_path": str(audio_path),
                "script": str(item.get("script", "")),
                "target_text": target_text,
                "emotion": str(item.get("emotion", "")),
                "messages": build_messages(
                    image_paths=[str(face_path)],
                    audio_path=str(audio_path),
                    script=str(item.get("script", "")),
                    target_text=target_text,
                    instruction=instruction,
                    system_prompt=system_prompt,
                    emotion=emotion,
                ),
            }
            records.append(record)

    write_jsonl(out_jsonl, records)
    print(f"saved: {out_jsonl}")
    print(f"generated: {len(records)}")
    print(f"attempted: {attempted}")
    return records


def main_meld() -> None:
    parser = argparse.ArgumentParser(description="Export MELD samples into Qwen2.5-Omni JSONL format.")
    parser.add_argument("--input_pt", type=str, required=True, help="Pseudo-summary MELD dataset (.pt)")
    parser.add_argument("--videos_root", type=str, required=True)
    parser.add_argument("--meld_csv", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--media_root", type=str, default="out/qwen_omni_meld_media")
    parser.add_argument("--num_frames", type=int, default=1, choices=[1, 3])
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--include_emotion_label", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    export_meld_manifest(
        input_pt=args.input_pt,
        videos_root=args.videos_root,
        meld_csv=args.meld_csv,
        out_jsonl=args.out_jsonl,
        media_root=args.media_root,
        num_frames=args.num_frames,
        instruction=args.instruction,
        system_prompt=args.system_prompt,
        include_emotion_label=args.include_emotion_label,
        limit=args.limit,
    )


def main_ko() -> None:
    parser = argparse.ArgumentParser(description="Export Korean samples into Qwen2.5-Omni JSONL format.")
    parser.add_argument("--json_path", type=str, default=str(env_path("MUTON_MULTI_TEXT_JSON", "preprocessing/multi_text.json")))
    parser.add_argument("--face_root", type=str, default=str(env_path("MUTON_FACE_ROOT", "data/face_crops")))
    parser.add_argument("--audio_root", type=str, default=str(env_path("MUTON_AUDIO_ROOT", "data/audio")))
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--include_emotion_label", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    export_ko_manifest(
        json_path=args.json_path,
        face_root=args.face_root,
        audio_root=args.audio_root,
        out_jsonl=args.out_jsonl,
        instruction=args.instruction,
        system_prompt=args.system_prompt,
        include_emotion_label=args.include_emotion_label,
        limit=args.limit,
    )
