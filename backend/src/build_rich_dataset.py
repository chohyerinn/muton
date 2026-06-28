from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from muton.config import env_path
from muton.encoders import AudioEncoder, FaceEncoder, TextEncoder


def time_str_to_seconds(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    s = value.strip()
    if not s:
        return 0.0
    try:
        hh, mm, rest = s.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
    except Exception:
        return 0.0


def load_meld_time_ranges(csv_path: str) -> Dict[str, Tuple[float, float]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {path}")

    ranges: Dict[str, Tuple[float, float]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                dialogue_id = int(str(row.get("Dialogue_ID", "")).strip())
                utterance_id = int(str(row.get("Utterance_ID", "")).strip())
            except ValueError:
                continue
            sample_id = f"meld_d{dialogue_id}_u{utterance_id}"
            ranges[sample_id] = (
                time_str_to_seconds(str(row.get("StartTime", ""))),
                time_str_to_seconds(str(row.get("EndTime", ""))),
            )
    return ranges


def parse_meld_video_path(videos_root: Path, sample_id: str) -> Path:
    prefix = "meld_d"
    if not sample_id.startswith(prefix) or "_u" not in sample_id:
        raise ValueError(f"Unexpected MELD sample id: {sample_id}")
    dialogue_str, utterance_str = sample_id[len(prefix):].split("_u", 1)
    return videos_root / f"dia{int(dialogue_str)}_utt{int(utterance_str)}.mp4"


def choose_timepoints(start_t: float, end_t: float, num_frames: int) -> List[float]:
    if not (end_t > start_t):
        return [max(0.0, start_t)]
    if num_frames == 1:
        return [(start_t + end_t) / 2.0]
    duration = end_t - start_t
    return [
        start_t + duration * 0.2,
        start_t + duration * 0.5,
        start_t + duration * 0.8,
    ]


def load_frame_at_time(video_path: Path, t_sec: float) -> Optional[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_MSEC, float(max(0.0, t_sec)) * 1000.0)
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        return None
    return frame_bgr


def extract_meld_frames(
    video_path: Path,
    start_t: float,
    end_t: float,
    num_frames: int,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for timestamp in choose_timepoints(start_t, end_t, num_frames):
        frame_bgr = load_frame_at_time(video_path, timestamp)
        if frame_bgr is not None:
            frames.append(frame_bgr)

    if not frames:
        frame_bgr = load_frame_at_time(video_path, 0.0)
        if frame_bgr is not None:
            frames.append(frame_bgr)
    return frames


def downsample_token_sequence(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    tokens = tokens.detach().cpu().float()
    if tokens.dim() != 2:
        raise ValueError(f"Expected [seq, dim] tokens, got shape {tuple(tokens.shape)}")
    if max_tokens <= 0 or tokens.size(0) <= max_tokens:
        return tokens.contiguous()

    pooled = F.adaptive_avg_pool1d(tokens.transpose(0, 1).unsqueeze(0), max_tokens)
    return pooled.squeeze(0).transpose(0, 1).contiguous()


def crop_face_bgr(face_encoder: FaceEncoder, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    img_h, img_w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_encoder.face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    aligned = face_encoder.align_face(frame_bgr, landmarks)

    x_list = [landmark.x for landmark in landmarks]
    y_list = [landmark.y for landmark in landmarks]
    x_min, x_max = min(x_list), max(x_list)
    y_min, y_max = min(y_list), max(y_list)

    cx = int((x_min + x_max) / 2 * img_w)
    cy = int((y_min + y_max) / 2 * img_h)
    w = int((x_max - x_min) * img_w)
    h = int((y_max - y_min) * img_h)

    padding = max(w, h) * 0.6
    x1 = max(0, int(cx - w / 2 - padding))
    y1 = max(0, int(cy - h / 2 - padding))
    x2 = min(img_w, int(cx + w / 2 + padding))
    y2 = min(img_h, int(cy + h / 2 + padding))

    face_bgr = aligned[y1:y2, x1:x2]
    if face_bgr is None or face_bgr.size == 0:
        return None
    if face_bgr.shape[0] < 10 or face_bgr.shape[1] < 10:
        return None
    return face_bgr


def save_face_debug(debug_dir: Path, sample_id: str, frame_bgr: np.ndarray, crop_bgr: Optional[np.ndarray]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{sample_id}_orig.jpg"), frame_bgr)
    if crop_bgr is None:
        (debug_dir / f"{sample_id}_crop_fail.txt").write_text("face_crop_failed\n", encoding="utf-8")
        return
    cv2.imwrite(str(debug_dir / f"{sample_id}_crop.jpg"), crop_bgr)


def extract_face_sequence(
    face_encoder: FaceEncoder,
    frame_bgr: np.ndarray,
    max_face_tokens: int,
    debug_dir: Optional[Path] = None,
    debug_id: str = "",
) -> Optional[Dict[str, torch.Tensor]]:
    crop_bgr = crop_face_bgr(face_encoder, frame_bgr)
    if debug_dir is not None:
        save_face_debug(debug_dir, debug_id, frame_bgr, crop_bgr)
    if crop_bgr is None:
        return None

    face_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    inputs = face_encoder.processor(images=face_rgb, return_tensors="pt").to(face_encoder.model.device)
    with torch.no_grad():
        outputs = face_encoder.model(**inputs, output_hidden_states=True, return_dict=True)

    hidden = outputs.hidden_states[-1].squeeze(0).detach().cpu().float()
    logits = outputs.logits.squeeze(0).detach().cpu().float()
    face_vec = hidden[0].clone()
    face_tokens = downsample_token_sequence(hidden[1:], max_face_tokens)

    return {
        "face_vec": face_vec,
        "face_emo_logits": logits,
        "face_tokens": face_tokens,
    }


def merge_face_sequences(face_parts: List[Dict[str, torch.Tensor]], max_face_tokens: int) -> Dict[str, torch.Tensor]:
    face_tokens = torch.cat([part["face_tokens"] for part in face_parts], dim=0)
    return {
        "face_vec": torch.stack([part["face_vec"] for part in face_parts], dim=0).mean(dim=0),
        "face_emo_logits": torch.stack([part["face_emo_logits"] for part in face_parts], dim=0).mean(dim=0),
        "face_tokens": downsample_token_sequence(face_tokens, max_face_tokens),
    }


def extract_audio_wav_with_ffmpeg(video_path: Path, wav_path: Path, sr: int = 16000) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(wav_path),
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:1000])


def extract_audio_sequence(
    audio_encoder: AudioEncoder,
    pcm_np: np.ndarray,
    max_audio_tokens: int,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        wav = torch.tensor(pcm_np, dtype=torch.float32, device=audio_encoder.device).unsqueeze(0)
        outputs = audio_encoder.wavlm(wav, output_hidden_states=True)
        hidden = outputs.last_hidden_state.squeeze(0)

        audio_tokens = downsample_token_sequence(hidden, max_audio_tokens)
        content = hidden.mean(dim=0).detach().cpu().float()

        seq_len = hidden.size(0)
        seconds = len(pcm_np) / audio_encoder.sample_rate
        frames_per_second = seq_len / seconds if seconds > 0 else seq_len
        last_n = max(1, min(seq_len, int(frames_per_second)))
        speaker = hidden[-last_n:].mean(dim=0).detach().cpu().float()

        wav_cpu = wav.squeeze(0).cpu()
        frame_len = int(0.025 * audio_encoder.sample_rate)
        hop = int(0.010 * audio_encoder.sample_rate)
        energies = [
            wav_cpu[i : i + frame_len].abs().mean().item()
            for i in range(0, wav_cpu.numel() - frame_len, hop)
        ]
        if not energies:
            energies = [1.0]

        x_old = np.linspace(0, 1, len(energies))
        x_new = np.linspace(0, 1, seq_len)
        energy_interp = np.interp(x_new, x_old, energies)
        weights = energy_interp / (energy_interp.sum() + 1e-9)
        weights_t = torch.tensor(weights, device=audio_encoder.device, dtype=torch.float32).unsqueeze(1)
        prosody = (hidden * weights_t).sum(dim=0).detach().cpu().float()

    return {
        "audio_tokens": audio_tokens,
        "audio_content": content,
        "audio_speaker": speaker,
        "audio_prosody": prosody,
    }


def extract_text_sequence(
    text_encoder: TextEncoder,
    text: str,
    max_text_tokens: int,
) -> Dict[str, torch.Tensor]:
    inputs = text_encoder.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=text_encoder.max_length,
    ).to(text_encoder.device)

    with torch.no_grad():
        outputs = text_encoder.model(**inputs, return_dict=True)

    last_hidden = outputs.last_hidden_state.squeeze(0)
    mask = inputs["attention_mask"].squeeze(0).bool()
    valid_tokens = last_hidden[mask]
    if valid_tokens.numel() == 0:
        valid_tokens = last_hidden[:1]

    mask_f = mask.unsqueeze(-1).float()
    pooled = ((last_hidden * mask_f).sum(dim=0) / mask_f.sum(dim=0).clamp(min=1e-6)).detach().cpu().float()

    return {
        "text": pooled,
        "text_tokens": downsample_token_sequence(valid_tokens, max_text_tokens),
    }


def get_summary_text(item: Dict[str, Any]) -> str:
    for key in ["summary_text", "summary_textc", "accessible_emotion_desc"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_rich_meld_dataset(
    input_pt: str,
    videos_root: str,
    meld_csv: str,
    out_pt: str,
    cache_dir: str,
    num_frames: int,
    max_face_tokens: int,
    max_audio_tokens: int,
    max_text_tokens: int,
    limit: int,
    debug_face_dir: str,
) -> List[Dict[str, Any]]:
    data = torch.load(input_pt, map_location="cpu")
    time_ranges = load_meld_time_ranges(meld_csv)

    videos_root_path = Path(videos_root)
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(debug_face_dir) if debug_face_dir else None

    face_encoder = FaceEncoder()
    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder()

    output: List[Dict[str, Any]] = []
    attempted = 0
    for sample in data:
        if limit > 0 and attempted >= limit:
            break
        attempted += 1

        target_text = str(sample.get("target_text", "")).strip()
        if not target_text:
            continue

        sample_id = str(sample.get("id", "")).strip()
        video_path = parse_meld_video_path(videos_root_path, sample_id)
        if not video_path.exists():
            print(f"[skip] missing video: {sample_id} -> {video_path}")
            continue

        start_t, end_t = time_ranges.get(sample_id, (0.0, 0.0))
        frames = extract_meld_frames(video_path, start_t, end_t, num_frames)
        if not frames:
            print(f"[skip] frame read fail: {sample_id} -> {video_path}")
            continue

        face_parts = []
        for frame_idx, frame_bgr in enumerate(frames):
            face_part = extract_face_sequence(
                face_encoder,
                frame_bgr,
                max_face_tokens=max_face_tokens,
                debug_dir=debug_dir,
                debug_id=f"{sample_id}_f{frame_idx}",
            )
            if face_part is not None:
                face_parts.append(face_part)

        if not face_parts:
            print(f"[skip] face crop fail: {sample_id}")
            continue

        wav_path = cache_dir_path / f"{sample_id}.wav"
        if not wav_path.exists():
            try:
                extract_audio_wav_with_ffmpeg(video_path, wav_path, sr=16000)
            except Exception as exc:
                print(f"[skip] audio extract fail: {sample_id} / {exc}")
                continue

        try:
            pcm, sample_rate = sf.read(str(wav_path))
            if getattr(pcm, "ndim", 1) > 1:
                pcm = pcm.mean(axis=1)
            if sample_rate != 16000:
                raise RuntimeError(f"Unexpected sample rate: {sample_rate}")
            audio_pack = extract_audio_sequence(audio_encoder, np.asarray(pcm, dtype=np.float32), max_audio_tokens)
        except Exception as exc:
            print(f"[skip] audio embedding fail: {sample_id} / {exc}")
            continue

        try:
            text_pack = extract_text_sequence(
                text_encoder,
                str(sample.get("script", "")),
                max_text_tokens=max_text_tokens,
            )
        except Exception as exc:
            print(f"[skip] text embedding fail: {sample_id} / {exc}")
            continue

        merged_face = merge_face_sequences(face_parts, max_face_tokens=max_face_tokens)
        item = dict(sample)
        item.update(
            {
                "face_vec": merged_face["face_vec"],
                "face_emo_logits": merged_face["face_emo_logits"],
                "face_tokens": merged_face["face_tokens"],
                "audio_content": audio_pack["audio_content"],
                "audio_speaker": audio_pack["audio_speaker"],
                "audio_prosody": audio_pack["audio_prosody"],
                "audio_tokens": audio_pack["audio_tokens"],
                "text": text_pack["text"],
                "text_tokens": text_pack["text_tokens"],
                "rich_num_frames": len(face_parts),
            }
        )
        output.append(item)

        if len(output) % 25 == 0:
            print(f"[ok] built {len(output)} samples (attempted={attempted})")

    torch.save(output, out_pt)
    print(f"saved: {out_pt}")
    print(f"generated: {len(output)}")
    print(f"attempted: {attempted}")
    return output


def build_rich_ko_dataset(
    json_path: str,
    face_root: str,
    audio_root: str,
    out_pt: str,
    max_face_tokens: int,
    max_audio_tokens: int,
    max_text_tokens: int,
    limit: int,
    debug_face_dir: str,
) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    face_root_path = Path(face_root)
    audio_root_path = Path(audio_root)
    debug_dir = Path(debug_face_dir) if debug_face_dir else None

    face_encoder = FaceEncoder()
    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder()

    output: List[Dict[str, Any]] = []
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
                torch.save(output, out_pt)
                print(f"saved: {out_pt}")
                print(f"generated: {len(output)}")
                print(f"attempted: {attempted}")
                return output

            attempted += 1
            sample_id = str(item.get("name", "")).strip()
            if not sample_id:
                continue

            summary = get_summary_text(item)
            if not summary:
                continue

            face_path = face_dir / f"{sample_id}.jpg"
            if not face_path.exists():
                face_path = face_dir / f"{sample_id}.jpeg"
            audio_path = audio_dir / f"{sample_id}.wav"
            if not face_path.exists() or not audio_path.exists():
                continue

            frame_bgr = cv2.imread(str(face_path))
            if frame_bgr is None:
                continue

            face_pack = extract_face_sequence(
                face_encoder,
                frame_bgr,
                max_face_tokens=max_face_tokens,
                debug_dir=debug_dir,
                debug_id=sample_id,
            )
            if face_pack is None:
                continue

            try:
                pcm, sample_rate = sf.read(str(audio_path))
                if getattr(pcm, "ndim", 1) > 1:
                    pcm = pcm.mean(axis=1)
                if sample_rate != 16000:
                    raise RuntimeError(f"Unexpected sample rate: {sample_rate}")
                audio_pack = extract_audio_sequence(audio_encoder, np.asarray(pcm, dtype=np.float32), max_audio_tokens)
            except Exception as exc:
                print(f"[skip] audio embedding fail: {sample_id} / {exc}")
                continue

            script = str(item.get("script", ""))
            text_pack = extract_text_sequence(text_encoder, script, max_text_tokens=max_text_tokens)

            output.append(
                {
                    "id": sample_id,
                    "person_id": person_id,
                    "face_vec": face_pack["face_vec"],
                    "face_emo_logits": face_pack["face_emo_logits"],
                    "face_tokens": face_pack["face_tokens"],
                    "audio_content": audio_pack["audio_content"],
                    "audio_speaker": audio_pack["audio_speaker"],
                    "audio_prosody": audio_pack["audio_prosody"],
                    "audio_tokens": audio_pack["audio_tokens"],
                    "text": text_pack["text"],
                    "text_tokens": text_pack["text_tokens"],
                    "emotion": item.get("emotion", "Neutral"),
                    "arousal": float(item.get("arousal", 5.0)),
                    "valence": float(item.get("valence", 5.0)),
                    "target_text": summary,
                    "script": script,
                }
            )

            if len(output) % 25 == 0:
                print(f"[ok] built {len(output)} samples (attempted={attempted})")

    torch.save(output, out_pt)
    print(f"saved: {out_pt}")
    print(f"generated: {len(output)}")
    print(f"attempted: {attempted}")
    return output


def main_meld() -> None:
    parser = argparse.ArgumentParser(description="Build a rich MELD dataset with sequence-level modality tokens.")
    parser.add_argument("--input_pt", type=str, required=True, help="Pseudo-summary MELD dataset (.pt)")
    parser.add_argument("--videos_root", type=str, required=True, help="Directory with dia{d}_utt{u}.mp4 files")
    parser.add_argument("--meld_csv", type=str, required=True, help="MELD CSV with StartTime/EndTime")
    parser.add_argument("--out_pt", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="out/rich_cache/meld_audio")
    parser.add_argument("--num_frames", type=int, default=1, choices=[1, 3])
    parser.add_argument("--max_face_tokens", type=int, default=96)
    parser.add_argument("--max_audio_tokens", type=int, default=128)
    parser.add_argument("--max_text_tokens", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug_face_dir", type=str, default="")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    build_rich_meld_dataset(
        input_pt=args.input_pt,
        videos_root=args.videos_root,
        meld_csv=args.meld_csv,
        out_pt=args.out_pt,
        cache_dir=args.cache_dir,
        num_frames=args.num_frames,
        max_face_tokens=args.max_face_tokens,
        max_audio_tokens=args.max_audio_tokens,
        max_text_tokens=args.max_text_tokens,
        limit=args.limit,
        debug_face_dir=args.debug_face_dir,
    )


def main_ko() -> None:
    parser = argparse.ArgumentParser(description="Build a rich Korean dataset with sequence-level modality tokens.")
    parser.add_argument("--json_path", type=str, default=str(env_path("MUTON_MULTI_TEXT_JSON", "preprocessing/multi_text.json")))
    parser.add_argument("--face_root", type=str, default=str(env_path("MUTON_FACE_ROOT", "data/face_crops")))
    parser.add_argument("--audio_root", type=str, default=str(env_path("MUTON_AUDIO_ROOT", "data/audio")))
    parser.add_argument("--out_pt", type=str, default=str(env_path("MUTON_FUSION_RICH_DATASET", "out/fusion_dataset_rich.pt")))
    parser.add_argument("--max_face_tokens", type=int, default=96)
    parser.add_argument("--max_audio_tokens", type=int, default=128)
    parser.add_argument("--max_text_tokens", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug_face_dir", type=str, default="")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    build_rich_ko_dataset(
        json_path=args.json_path,
        face_root=args.face_root,
        audio_root=args.audio_root,
        out_pt=args.out_pt,
        max_face_tokens=args.max_face_tokens,
        max_audio_tokens=args.max_audio_tokens,
        max_text_tokens=args.max_text_tokens,
        limit=args.limit,
        debug_face_dir=args.debug_face_dir,
    )
