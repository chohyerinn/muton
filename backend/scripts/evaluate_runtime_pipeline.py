from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import requests


TEXT_CLEAN_RE = re.compile(r"\s+")


def normalize_for_cer(text: str) -> str:
    return TEXT_CLEAN_RE.sub("", text.strip().lower())


def normalize_for_wer(text: str) -> list[str]:
    return [token for token in TEXT_CLEAN_RE.sub(" ", text.strip().lower()).split(" ") if token]


def levenshtein(a: list[Any] | str, b: list[Any] | str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref or not hyp:
        return 0.0
    lcs = lcs_len(ref, hyp)
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def infer_audio_format(path: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit.lower()
    if path.suffix.lower() == ".wav":
        return "wav"
    return "pcm"


def wav_to_pcm_bytes(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path} is not 16-bit PCM WAV.")
    if sample_rate != 16000:
        raise ValueError(f"{path} must be 16kHz for the current MUTON runtime, got {sample_rate}Hz.")

    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return pcm.tobytes()


def read_audio_as_pcm(path: Path, audio_format: str) -> bytes:
    if audio_format == "wav":
        return wav_to_pcm_bytes(path)
    if audio_format == "pcm":
        return path.read_bytes()
    raise ValueError(f"Unsupported audio format: {audio_format}")


def resolve_path(manifest_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path


def post_timed(method: str, url: str, **kwargs: Any) -> tuple[dict[str, Any], float, int]:
    started = time.perf_counter()
    response = requests.request(method, url, **kwargs)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.json(), elapsed, response.status_code


def send_audio_chunks(
    base_url: str,
    pcm_bytes: bytes,
    *,
    chunk_bytes: int,
    flush_silence_ms: int,
    timeout: int,
) -> tuple[str, float, list[float], list[dict[str, Any]]]:
    endpoint = f"{base_url.rstrip('/')}/process_audio_chunk"
    chunk_latencies: list[float] = []
    responses: list[dict[str, Any]] = []
    final_text = ""
    final_confidence = 0.0

    chunks = [pcm_bytes[i : i + chunk_bytes] for i in range(0, len(pcm_bytes), chunk_bytes)]
    silence_bytes = b"\x00" * int(16000 * 2 * (flush_silence_ms / 1000.0))
    chunks.extend(silence_bytes[i : i + chunk_bytes] for i in range(0, len(silence_bytes), chunk_bytes))

    for index, chunk in enumerate(chunks):
        if not chunk:
            continue
        result, elapsed, _ = post_timed(
            "POST",
            endpoint,
            files={"audio": (f"chunk_{index:04d}.pcm", chunk, "application/octet-stream")},
            timeout=timeout,
        )
        chunk_latencies.append(elapsed)
        responses.append(result)
        text = str(result.get("text", "") or "").strip()
        if text:
            final_text = text
            final_confidence = float(result.get("stt_confidence", 0.0) or 0.0)

    return final_text, final_confidence, chunk_latencies, responses


def post_eval_summary(
    base_url: str,
    *,
    text: str,
    mode: str,
    use_adapter: bool,
    image_path: Path | None,
    pcm_bytes: bytes | None,
    timeout: int,
) -> tuple[dict[str, Any], float]:
    files: dict[str, tuple[str, bytes, str]] = {}
    if image_path is not None and mode in {"full", "text_face"}:
        files["frame"] = (image_path.name, image_path.read_bytes(), "image/jpeg")
    if pcm_bytes is not None and mode in {"full", "text_audio"}:
        files["audio"] = ("sample.pcm", pcm_bytes, "application/octet-stream")

    result, elapsed, _ = post_timed(
        "POST",
        f"{base_url.rstrip('/')}/eval/generate_summary",
        data={
            "text": text,
            "mode": mode,
            "use_adapter": str(use_adapter).lower(),
            "audio_format": "pcm",
        },
        files=files,
        timeout=timeout,
    )
    return result, elapsed


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize_report(stt_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# MUTON Evaluation Report",
        "",
        "## STT",
        "",
    ]
    if stt_rows:
        lines.extend(
            [
                f"- samples: {len(stt_rows)}",
                f"- mean CER: {mean([float(row.get('cer', 0.0) or 0.0) for row in stt_rows]):.4f}",
                f"- mean WER: {mean([float(row.get('wer', 0.0) or 0.0) for row in stt_rows]):.4f}",
                f"- mean total STT latency sec: {mean([float(row.get('stt_total_latency_sec', 0.0) or 0.0) for row in stt_rows]):.4f}",
                "",
            ]
        )
    else:
        lines.extend(["- skipped", ""])

    lines.extend(["## Summary", ""])
    if summary_rows:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in summary_rows:
            grouped.setdefault((str(row.get("mode")), str(row.get("variant"))), []).append(row)
        for (mode, variant), rows in sorted(grouped.items()):
            rouge_values = [float(row.get("rouge_l_f1", 0.0) or 0.0) for row in rows]
            latency_values = [float(row.get("summary_latency_sec", 0.0) or 0.0) for row in rows]
            lines.append(
                f"- {variant} / {mode}: n={len(rows)}, mean ROUGE-L F1={mean(rouge_values):.4f}, "
                f"mean latency={mean(latency_values):.4f}s"
            )
        lines.append("")
    else:
        lines.extend(["- skipped", ""])

    lines.extend(
        [
            "## Human Rating Columns",
            "",
            "Use `summary_human_eval_template.csv` for 1-5 human scoring:",
            "",
            "- emotion_reflection_1_5",
            "- intent_reflection_1_5",
            "- fluency_1_5",
            "- faithfulness_1_5",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MUTON runtime STT, summary, ablation, and latency.")
    parser.add_argument("--base_url", required=True, help="Example: http://127.0.0.1:5000")
    parser.add_argument("--manifest", type=Path, required=True, help="Evaluation manifest JSON")
    parser.add_argument("--output_dir", type=Path, default=Path("out/eval_runtime"))
    parser.add_argument("--chunk_bytes", type=int, default=32000)
    parser.add_argument("--flush_silence_ms", type=int, default=1200)
    parser.add_argument("--summary_modes", default="text,text_face,text_audio,full")
    parser.add_argument("--summary_variants", default="lora,base")
    parser.add_argument("--skip_stt", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    manifest_dir = args.manifest.resolve().parent
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest must contain a non-empty samples list.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stt_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []

    health, health_latency, _ = post_timed("GET", f"{args.base_url.rstrip('/')}/health", timeout=30)
    print(f"[health] {health} ({health_latency:.3f}s)")

    modes = [item.strip() for item in args.summary_modes.split(",") if item.strip()]
    variants = [item.strip() for item in args.summary_variants.split(",") if item.strip()]

    for sample in samples:
        sample_id = str(sample.get("id") or f"sample_{len(raw_results) + 1}")
        image_path = resolve_path(manifest_dir, sample.get("image") or sample.get("image_path"))
        audio_path = resolve_path(manifest_dir, sample.get("audio") or sample.get("audio_path"))
        audio_format = infer_audio_format(audio_path, sample.get("audio_format")) if audio_path else ""
        reference_text = str(sample.get("reference_text", "") or "")
        reference_summary = str(sample.get("reference_summary", "") or "")

        pcm_bytes = read_audio_as_pcm(audio_path, audio_format) if audio_path else None
        predicted_text = ""
        stt_confidence = 0.0
        sample_raw: dict[str, Any] = {"id": sample_id, "raw_stt_responses": []}

        if not args.skip_stt and pcm_bytes is not None:
            print(f"[stt] {sample_id}")
            started = time.perf_counter()
            predicted_text, stt_confidence, chunk_latencies, raw_stt = send_audio_chunks(
                args.base_url,
                pcm_bytes,
                chunk_bytes=args.chunk_bytes,
                flush_silence_ms=args.flush_silence_ms,
                timeout=args.timeout,
            )
            total_latency = time.perf_counter() - started
            sample_raw["raw_stt_responses"] = raw_stt
            stt_rows.append(
                {
                    "sample_id": sample_id,
                    "reference_text": reference_text,
                    "predicted_text": predicted_text,
                    "stt_confidence": stt_confidence,
                    "cer": cer(reference_text, predicted_text) if reference_text else "",
                    "wer": wer(reference_text, predicted_text) if reference_text else "",
                    "chunk_count": len(chunk_latencies),
                    "mean_chunk_latency_sec": round(mean(chunk_latencies), 4),
                    "max_chunk_latency_sec": round(max(chunk_latencies) if chunk_latencies else 0.0, 4),
                    "stt_total_latency_sec": round(total_latency, 4),
                }
            )

        summary_text_input = str(sample.get("summary_text", "") or "").strip() or reference_text.strip() or predicted_text.strip()
        if not args.skip_summary and summary_text_input:
            for mode in modes:
                if mode in {"full", "text_face"} and image_path is None:
                    continue
                if mode in {"full", "text_audio"} and pcm_bytes is None:
                    continue

                for variant in variants:
                    use_adapter = variant.lower() not in {"base", "no_lora", "without_lora"}
                    print(f"[summary] {sample_id} mode={mode} variant={variant}")
                    try:
                        result, elapsed = post_eval_summary(
                            args.base_url,
                            text=summary_text_input,
                            mode=mode,
                            use_adapter=use_adapter,
                            image_path=image_path,
                            pcm_bytes=pcm_bytes,
                            timeout=args.timeout,
                        )
                        generated_summary = str(result.get("summary", "") or "")
                        error = ""
                    except Exception as exc:
                        result = {}
                        elapsed = 0.0
                        generated_summary = ""
                        error = str(exc)

                    row = {
                        "sample_id": sample_id,
                        "mode": mode,
                        "variant": variant,
                        "use_adapter": use_adapter,
                        "input_text": summary_text_input,
                        "reference_summary": reference_summary,
                        "generated_summary": generated_summary,
                        "rouge_l_f1": rouge_l_f1(reference_summary, generated_summary) if reference_summary else "",
                        "summary_latency_sec": round(float(result.get("latency_sec", elapsed) or elapsed), 4),
                        "error": error,
                    }
                    summary_rows.append(row)
                    human_rows.append(
                        {
                            **row,
                            "emotion_reflection_1_5": "",
                            "intent_reflection_1_5": "",
                            "fluency_1_5": "",
                            "faithfulness_1_5": "",
                            "notes": "",
                        }
                    )

        raw_results.append(sample_raw)

    write_csv(args.output_dir / "stt_results.csv", stt_rows)
    write_csv(args.output_dir / "summary_results.csv", summary_rows)
    write_csv(args.output_dir / "summary_human_eval_template.csv", human_rows)
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "health": health,
                "stt": stt_rows,
                "summary": summary_rows,
                "raw": raw_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(summarize_report(stt_rows, summary_rows), encoding="utf-8")
    print(f"[done] wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
