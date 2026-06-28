from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def post_video(base_url: str, image_path: Path) -> dict:
    with open(image_path, "rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/process_video_chunk",
            files={"frame": (image_path.name, handle, "image/jpeg")},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()


def post_audio(base_url: str, pcm_path: Path) -> dict:
    with open(pcm_path, "rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/process_audio_chunk",
            files={"audio": (pcm_path.name, handle, "application/octet-stream")},
            timeout=60,
        )
    response.raise_for_status()
    return response.json()


def post_summary(base_url: str, text: str) -> dict:
    response = requests.post(
        f"{base_url.rstrip('/')}/get_fusion_analysis",
        data={
            "text": text,
            "prosody": "[]",
            "content": "[]",
            "speaker": "[]",
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MUTON API client example.")
    parser.add_argument("--base_url", required=True, help="Example: http://127.0.0.1:5000")
    parser.add_argument("--image", type=Path, required=True, help="JPEG frame path")
    parser.add_argument("--pcm", type=Path, required=True, help="16kHz mono int16 PCM path")
    parser.add_argument("--text", type=str, default="", help="Optional transcript text for summary request")
    args = parser.parse_args()

    video_result = post_video(args.base_url, args.image)
    print("[video]")
    print(json.dumps(video_result, ensure_ascii=False, indent=2))

    audio_result = post_audio(args.base_url, args.pcm)
    print("[audio]")
    print(json.dumps(audio_result, ensure_ascii=False, indent=2))

    summary_text = args.text.strip() or audio_result.get("text", "")
    if summary_text:
        summary_result = post_summary(args.base_url, summary_text)
        print("[summary]")
        print(json.dumps(summary_result, ensure_ascii=False, indent=2))
    else:
        print("[summary]")
        print("Skipped because no finalized transcript was returned.")


if __name__ == "__main__":
    main()
