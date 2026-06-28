from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


TRANSLATE_SYSTEM = (
    "You are a professional translator for dialogue. Translate the English conversational utterance "
    "into natural Korean spoken style. Preserve emotion, intensity, sarcasm, and register. "
    "Do not add explanations. Output only the Korean translation."
)

SUMMARY_SYSTEM = (
    "You write Korean reference summaries for a multimodal dialogue-assistance benchmark. "
    "Given an utterance, MELD emotion, and sentiment, write exactly one concise Korean sentence "
    "describing the speaker's emotion, tone, intent, and situation. Do not add bullets or labels."
)


def time_str_to_seconds(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    text = value.strip()
    if not text:
        return 0.0
    try:
        hh, mm, rest = text.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
    except Exception:
        return 0.0


def normalize_emotion(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "anger": "angry",
        "angry": "angry",
        "sadness": "sad",
        "sad": "sad",
        "neutral": "neutral",
        "joy": "happy",
        "happy": "happy",
        "surprise": "surprise",
        "fear": "fear",
        "disgust": "disgust",
    }
    return mapping.get(text, text or "neutral")


def load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def call_chat(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    max_retries: int = 5,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            text = response.choices[0].message.content or ""
            return text.strip().strip('"').strip("'").strip()
        except Exception as exc:
            wait = (2**attempt) * 0.8
            print(f"[retry {attempt + 1}/{max_retries}] {exc} -> sleep {wait:.1f}s")
            time.sleep(wait)
    return ""


def translate_utterance(
    client: Any | None,
    cache: dict[str, Any],
    *,
    sample_id: str,
    utterance_en: str,
    model: str,
    enabled: bool,
) -> str:
    cache_key = f"translation:{sample_id}"
    if cache_key in cache:
        return str(cache[cache_key])
    if not enabled or client is None:
        translated = utterance_en
    else:
        translated = call_chat(client, model=model, system=TRANSLATE_SYSTEM, user=utterance_en) or utterance_en
    cache[cache_key] = translated
    return translated


def build_reference_summary(
    client: Any | None,
    cache: dict[str, Any],
    *,
    sample_id: str,
    script_ko: str,
    emotion: str,
    sentiment: str,
    model: str,
    enabled: bool,
) -> str:
    cache_key = f"summary:{sample_id}"
    if cache_key in cache:
        return str(cache[cache_key])

    if not enabled or client is None:
        summary = f"화자가 {emotion} 감정으로 '{script_ko}'라고 말하는 상황이다."
    else:
        prompt = (
            f"Utterance in Korean: {script_ko}\n"
            f"MELD emotion: {emotion}\n"
            f"MELD sentiment: {sentiment}\n"
        )
        summary = call_chat(client, model=model, system=SUMMARY_SYSTEM, user=prompt)
        if not summary:
            summary = f"화자가 {emotion} 감정으로 '{script_ko}'라고 말하는 상황이다."

    cache[cache_key] = summary
    return summary


def load_frame_at_time(video_path: Path, t_sec: float) -> Any | None:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def save_frame(video_path: Path, image_path: Path, t_sec: float) -> bool:
    import cv2

    frame = load_frame_at_time(video_path, t_sec)
    if frame is None and t_sec != 0.0:
        frame = load_frame_at_time(video_path, 0.0)
    if frame is None:
        return False
    image_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(image_path), frame))


def extract_audio(video_path: Path, wav_path: Path) -> bool:
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
        "16000",
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"[skip] ffmpeg failed for {video_path.name}: {proc.stderr.strip()[:500]}")
        return False
    return True


def pick_rows(df: Any, max_samples: int, emotion_balanced: bool) -> Any:
    if max_samples <= 0 or max_samples >= len(df):
        return df
    if not emotion_balanced or "Emotion" not in df.columns:
        return df.head(max_samples)

    buckets: dict[str, list[int]] = {}
    for index, row in df.iterrows():
        buckets.setdefault(normalize_emotion(row.get("Emotion")), []).append(index)

    selected: list[int] = []
    while len(selected) < max_samples:
        added = False
        for emotion in sorted(buckets):
            if buckets[emotion] and len(selected) < max_samples:
                selected.append(buckets[emotion].pop(0))
                added = True
        if not added:
            break
    return df.loc[selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MUTON runtime evaluation samples from MELD CSV and Raw mp4 files.")
    parser.add_argument("--csv", type=Path, required=True, help="MELD CSV such as train_sent_emo.csv")
    parser.add_argument("--videos_root", type=Path, required=True, help="Directory containing dia{Dialogue_ID}_utt{Utterance_ID}.mp4")
    parser.add_argument("--output_dir", type=Path, default=Path("out/eval_meld_samples"))
    parser.add_argument("--manifest", type=Path, default=Path("out/eval_manifest_meld.json"))
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--emotion_balanced", action="store_true")
    parser.add_argument("--translate", action="store_true", help="Use OpenAI to translate utterances to Korean")
    parser.add_argument("--make_reference_summary", action="store_true", help="Use OpenAI to create Korean reference summaries")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--cache", type=Path, default=Path("out/eval_meld_cache.json"))
    args = parser.parse_args()

    import pandas as pd
    from openai import OpenAI

    df = pd.read_csv(args.csv)
    rows = pick_rows(df, args.max_samples, args.emotion_balanced)
    cache = load_json(args.cache)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    client = OpenAI(api_key=api_key) if api_key and (args.translate or args.make_reference_summary) else None
    if (args.translate or args.make_reference_summary) and client is None:
        print("[warn] OPENAI_API_KEY is not set. Text fields will use fallbacks.")

    samples: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        try:
            did = int(row["Dialogue_ID"])
            uid = int(row["Utterance_ID"])
        except Exception:
            continue

        sample_id = f"meld_d{did}_u{uid}"
        video_path = args.videos_root / f"dia{did}_utt{uid}.mp4"
        if not video_path.exists():
            print(f"[skip] missing video: {video_path}")
            continue

        start_t = time_str_to_seconds(row.get("StartTime", ""))
        end_t = time_str_to_seconds(row.get("EndTime", ""))
        mid_t = (start_t + end_t) / 2.0 if end_t > start_t else 0.0

        image_path = args.output_dir / "frames" / f"{sample_id}.jpg"
        audio_path = args.output_dir / "audio" / f"{sample_id}.wav"
        if not save_frame(video_path, image_path, mid_t):
            print(f"[skip] frame failed: {sample_id}")
            continue
        if not extract_audio(video_path, audio_path):
            continue

        utterance_en = str(row.get("Utterance", "") or "").strip()
        emotion = normalize_emotion(row.get("Emotion"))
        sentiment = str(row.get("Sentiment", "") or "").strip().lower() or "neutral"
        script_ko = translate_utterance(
            client,
            cache,
            sample_id=sample_id,
            utterance_en=utterance_en,
            model=args.openai_model,
            enabled=args.translate,
        )
        reference_summary = build_reference_summary(
            client,
            cache,
            sample_id=sample_id,
            script_ko=script_ko,
            emotion=emotion,
            sentiment=sentiment,
            model=args.openai_model,
            enabled=args.make_reference_summary,
        )

        samples.append(
            {
                "id": sample_id,
                "image_path": str(image_path.resolve()),
                "audio_path": str(audio_path.resolve()),
                "audio_format": "wav",
                "reference_text": script_ko,
                "reference_summary": reference_summary,
                "meld_utterance_en": utterance_en,
                "meld_emotion": emotion,
                "meld_sentiment": sentiment,
                "meld_dialogue_id": did,
                "meld_utterance_id": uid,
            }
        )
        print(f"[ok] {sample_id}")
        save_json(args.cache, cache)

    manifest = {"samples": samples}
    save_json(args.manifest, manifest)
    save_json(args.cache, cache)
    print(f"Saved manifest: {args.manifest}")
    print(f"N = {len(samples)}")


if __name__ == "__main__":
    main()
