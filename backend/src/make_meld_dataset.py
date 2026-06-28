# -*- coding: utf-8 -*-
"""
make_meld_dataset.py
- MELD Raw(mp4) + CSV(train_sent_emo.csv) -> MUTON fusion 학습용 .pt 생성
- encoders.py 기준:
  - FaceEncoder.encode_frame(frame_bgr) -> dict or None
  - AudioEncoder.extract_features_from_pcm(pcm_np_float32) -> dict(prosody/content/speaker)
  - TextEncoder.encode(text) -> np.ndarray(768,)
- OpenAI 번역은 (face + audio) 성공 샘플만 수행 + 캐시(JSON)
"""

from __future__ import annotations

import os
import json
import time
import argparse
import subprocess
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import cv2

# src/encoders.py
from muton.encoders import FaceEncoder, AudioEncoder, TextEncoder
from openai import OpenAI

def sentiment_to_valence(s: str) -> float:
    if not isinstance(s, str):
        return 5.0
    x = s.strip().lower()
    if x == "negative":
        return 2.0
    if x == "positive":
        return 8.0
    return 5.0

def emotion_to_arousal(e: str) -> float:
    m = {
        "Angry": 8.0, "Surprise": 7.5, "Fear": 7.0,
        "Happy": 6.5, "Disgust": 6.0,
        "Neutral": 5.0, "Sad": 3.5,
    }
    return float(m.get(e, 5.0))

# -----------------------
# Utils: time parsing
# -----------------------
def time_str_to_seconds(t: Any) -> float:
    """
    MELD StartTime/EndTime: 'HH:MM:SS,mmm' e.g. '00:16:16,059'
    """
    if not isinstance(t, str):
        return 0.0
    s = t.strip()
    if not s:
        return 0.0
    try:
        hh, mm, rest = s.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
    except Exception:
        return 0.0


# -----------------------
# Utils: video frame
# -----------------------
def load_frame_at_time(video_path: str, t_sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


# -----------------------
# Utils: audio extraction
# -----------------------
def extract_audio_wav_with_ffmpeg(video_path: str, wav_path: str, sr: int = 16000) -> None:
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "wav",
        wav_path
    ]
    # stderr를 숨기면 디버깅 지옥이므로, 실패 시 출력하도록 처리
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:1000])


def read_wav_mono_16k_float32(wav_path: str) -> np.ndarray:
    """
    ffmpeg로 만든 wav(16k, mono, s16le)를 numpy float32 [-1,1]로 읽기
    """
    import wave
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if fr != 16000:
        # ffmpeg에서 16k로 뽑으니 보통 안 걸림
        raise RuntimeError(f"Unexpected sample rate: {fr} (expected 16000)")
    if n_channels != 1:
        raise RuntimeError(f"Unexpected channels: {n_channels} (expected 1)")
    if sampwidth != 2:
        raise RuntimeError(f"Unexpected sampwidth: {sampwidth} (expected 2 bytes int16)")

    pcm_int16 = np.frombuffer(raw, dtype=np.int16)
    pcm = pcm_int16.astype(np.float32) / 32768.0
    return pcm


# -----------------------
# Translation cache + OpenAI
# -----------------------
TRANSLATE_SYSTEM = (
    "You are a professional translator for dialogue. "
    "Translate the English conversational utterance into natural Korean spoken style. "
    "Preserve emotion, intensity, sarcasm, and register. "
    "Do NOT add explanations. Output ONLY the Korean translation."
)

def load_json(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(path: str, obj: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def translate_en_to_ko_openai(
    client: OpenAI,
    text_en: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 5,
) -> str:
    t = (text_en or "").strip()
    if not t:
        return ""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TRANSLATE_SYSTEM},
                    {"role": "user", "content": t},
                ],
                temperature=0.2,
            )
            ko = resp.choices[0].message.content.strip()
            ko = ko.strip().strip('"').strip("'").strip()
            return ko
        except Exception as e:
            wait = (2 ** attempt) * 0.8
            print(f"[translate retry {attempt+1}/{max_retries}] {e} -> sleep {wait:.1f}s")
            time.sleep(wait)

    # 최후 fallback: 영어 그대로
    return t

def get_korean_script(
    sample_id: str,
    utterance_en: str,
    cache: dict,
    client: Optional[OpenAI],
    translate_model: str,
    do_translate: bool,
) -> str:
    if sample_id in cache:
        return cache[sample_id]

    if (not do_translate) or (client is None):
        ko = utterance_en
    else:
        ko = translate_en_to_ko_openai(client, utterance_en, model=translate_model)

    cache[sample_id] = ko
    return ko


# -----------------------
# Label mapping (MELD emotion -> MUTON emotion)
# -----------------------
def normalize_emotion(e: str) -> str:
    if not isinstance(e, str):
        return "Neutral"
    x = e.strip().lower()
    mapping = {
        "anger": "Angry",
        "angry": "Angry",
        "sadness": "Sad",
        "sad": "Sad",
        "neutral": "Neutral",
        "joy": "Happy",
        "happy": "Happy",
        "surprise": "Surprise",
        "fear": "Fear",
        "disgust": "Disgust",
    }
    return mapping.get(x, x.capitalize() if x else "Neutral")


# -----------------------
# (Optional) Face crop debug (실제 크롭 이미지 저장)
# - FaceEncoder.encode_frame 내부와 동일한 로직을 여기서 다시 돌려서 crop 저장
# -----------------------
def save_face_crop_debug(face_encoder: FaceEncoder, frame_bgr: np.ndarray, out_dir: str, sample_id: str) -> bool:
    """
    encoders.FaceEncoder 내부 crop 로직을 최대한 동일하게 재현해서
    orig + crop 이미지를 저장한다.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{sample_id}_orig.jpg"), frame_bgr)

        img_h, img_w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = face_encoder.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return False
        landmarks = results.multi_face_landmarks[0].landmark

        aligned = face_encoder.align_face(frame_bgr, landmarks)

        x_list = [l.x for l in landmarks]
        y_list = [l.y for l in landmarks]
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
            return False

        cv2.imwrite(os.path.join(out_dir, f"{sample_id}_crop.jpg"), face_bgr)
        return True
    except Exception:
        return False


# -----------------------
# Dataset builder
# -----------------------
def build_dataset(
    csv_path: str,
    videos_root: str,
    out_cache_dir: str,
    max_samples: int = -1,
    translate_cache_path: Optional[str] = None,
    translate_model: str = "gpt-4o-mini",
    do_translate: bool = True,
    debug_face_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    os.makedirs(out_cache_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # encoders
    face_encoder = FaceEncoder()
    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder()

    # openai translator
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    client = OpenAI(api_key=api_key) if (do_translate and api_key) else None
    if do_translate and client is None:
        print("⚠️ OPENAI_API_KEY not set. Translation will fallback to English.")

    translate_cache = load_json(translate_cache_path) if translate_cache_path else {}
    dataset: List[Dict[str, Any]] = []

    n_total = len(df) if max_samples is None or max_samples < 0 else min(len(df), max_samples)

    for idx in range(n_total):
        row = df.iloc[idx]

        did = int(row["Dialogue_ID"])
        uid = int(row["Utterance_ID"])
        sample_id = f"meld_d{did}_u{uid}"

        video_path = os.path.join(videos_root, f"dia{did}_utt{uid}.mp4")
        if not os.path.exists(video_path):
            print(f"[skip] video missing: {sample_id} -> {video_path}")
            continue

        utterance_en = str(row.get("Utterance", ""))
        speaker = str(row.get("Speaker", ""))
        emotion = normalize_emotion(str(row.get("Emotion", "Neutral")))

        # 1) frame at mid time
        start_t = time_str_to_seconds(row.get("StartTime", ""))
        end_t = time_str_to_seconds(row.get("EndTime", ""))
        mid_t = 0.0 if (not np.isfinite(start_t) or not np.isfinite(end_t) or end_t <= start_t) else (start_t + end_t) / 2.0

        frame = load_frame_at_time(video_path, mid_t)
        if frame is None:
            frame = load_frame_at_time(video_path, 0.0)
        if frame is None:
            print(f"[skip] frame read fail: {sample_id}")
            continue

        # 2) face first (filter)
        face_out = face_encoder.encode_frame(frame)
        if face_out is None or face_out.get("status") != "ok":
            print(f"[skip] face fail: {sample_id}")
            continue

        # (optional) save face crop debug
        if debug_face_dir:
            _ = save_face_crop_debug(face_encoder, frame, debug_face_dir, sample_id)

        face_vec = np.array(face_out["face_vec"], dtype=np.float32)
        face_emo_logits = np.array(face_out["face_emotion_logits"], dtype=np.float32)

        # ===== 얼굴 품질 필터 =====
        if np.linalg.norm(face_vec) < 1e-2:
            print(f"[skip] weak/empty face embedding: {sample_id}")
            continue

        if np.allclose(face_emo_logits, 0.0):
            print(f"[skip] empty face logits: {sample_id}")
            continue




        # 3) wav extract (filter)
        wav_path = os.path.join(out_cache_dir, f"{sample_id}.wav")
        if not os.path.exists(wav_path):
            try:
                extract_audio_wav_with_ffmpeg(video_path, wav_path, sr=16000)
            except Exception as e:
                print(f"[skip] audio extract fail: {sample_id} / {e}")
                continue

        # 4) audio embedding (filter)
        try:
            pcm = read_wav_mono_16k_float32(wav_path)
            audio_out = audio_encoder.extract_features_from_pcm(pcm)
            audio_content = np.array(audio_out["content"], dtype=np.float32)
            audio_speaker = np.array(audio_out["speaker"], dtype=np.float32)
            audio_prosody = np.array(audio_out["prosody"], dtype=np.float32)
        except Exception as e:
            print(f"[skip] audio embedding fail: {sample_id} / {e}")
            continue

        # 5) translate + text embedding (COST) - do after face+audio success
        try:
            script_ko = get_korean_script(
                sample_id=sample_id,
                utterance_en=utterance_en,
                cache=translate_cache,
                client=client,
                translate_model=translate_model,
                do_translate=do_translate,
            )
            text_vec = text_encoder.encode(script_ko)
        except Exception as e:
            print(f"[skip] text/translate fail: {sample_id} / {e}")
            continue

        # defaults (MELD에 arousal/valence 없음)
        sentiment = str(row.get("Sentiment", "neutral"))
        valence = sentiment_to_valence(sentiment)
        arousal = emotion_to_arousal(emotion)

        # target_text는 일단 script_ko를 그대로 두거나, 네 요약 생성 타깃이 있으면 거기로 바꿔라
        target_text = script_ko

        dataset.append({
            "id": sample_id,
            "person_id": speaker,  # speaker를 person_id로 사용
            "face_vec": face_vec,
            "face_emo_logits": face_emo_logits,
            "audio_content": audio_content,
            "audio_speaker": audio_speaker,
            "audio_prosody": audio_prosody,
            "text": text_vec,
            "emotion": emotion,
            "arousal": float(arousal),
            "valence": float(valence),
            "target_text": target_text,
            "script": script_ko,
        })

        if (len(dataset) % 50) == 0:
            print(f"[ok] built {len(dataset)} samples (idx={idx+1}/{n_total})")
            if translate_cache_path:
                save_json(translate_cache_path, translate_cache)

    if translate_cache_path:
        save_json(translate_cache_path, translate_cache)

    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--videos_root", type=str, required=True)
    ap.add_argument("--out_pt", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--max_samples", type=int, default=-1)

    ap.add_argument("--translate", action="store_true", help="Enable OpenAI translation (default on)")
    ap.add_argument("--no_translate", action="store_true", help="Disable translation; keep English")
    ap.add_argument("--translate_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--translate_cache", type=str, default="", help="path to json cache")

    ap.add_argument("--debug_face_dir", type=str, default="", help="if set, save orig+crop face images here")

    args = ap.parse_args()

    do_translate = True
    if args.no_translate:
        do_translate = False
    elif args.translate:
        do_translate = True

    out_pt_dir = os.path.dirname(args.out_pt)
    if out_pt_dir:
        os.makedirs(out_pt_dir, exist_ok=True)

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    translate_cache_path = args.translate_cache.strip()
    if not translate_cache_path:
        # cache_dir 안에 기본 캐시 파일 생성
        translate_cache_path = os.path.join(cache_dir, "meld_translate_cache.json")

    debug_face_dir = args.debug_face_dir.strip() or None

    ds = build_dataset(
        csv_path=args.csv,
        videos_root=args.videos_root,
        out_cache_dir=cache_dir,
        max_samples=args.max_samples,
        translate_cache_path=translate_cache_path,
        translate_model=args.translate_model,
        do_translate=do_translate,
        debug_face_dir=debug_face_dir,
    )

    torch.save(ds, args.out_pt)
    print(f"Saved: {args.out_pt}")
    print(f"N = {len(ds)}")

    if len(ds) > 0:
        first = ds[0]
        print("Keys:", list(first.keys()))
        print("face_vec:", np.array(first["face_vec"]).shape, "audio_content:", np.array(first["audio_content"]).shape, "text:", np.array(first["text"]).shape)


if __name__ == "__main__":
    main()
