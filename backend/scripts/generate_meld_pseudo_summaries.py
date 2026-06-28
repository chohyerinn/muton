import argparse
import base64
import csv
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image


MELD_ID_RE = re.compile(r"meld_d(?P<dialogue>\d+)_u(?P<utterance>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pseudo summary targets for MELD using translated scripts and representative video frames.",
    )
    parser.add_argument("--input_pt", type=str, required=True, help="Source MELD .pt file")
    parser.add_argument("--videos_root", type=str, required=True, help="Directory that contains dia{d}_utt{u}.mp4")
    parser.add_argument("--output_pt", type=str, required=True, help="Destination .pt with pseudo target_text")
    parser.add_argument("--meld_csv", type=str, default="", help="Optional MELD CSV with StartTime/EndTime for timestamp-aligned frame extraction")
    parser.add_argument("--cache_json", type=str, default="", help="Optional cache file to resume generation")
    parser.add_argument("--style_examples_pt", type=str, default="out/fusion_dataset.pt")
    parser.add_argument("--style_examples", type=int, default=4)
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "transformers"])
    parser.add_argument("--model", type=str, required=True, help="Model id for the selected backend")
    parser.add_argument("--base_url", type=str, default="", help="Optional OpenAI-compatible base URL")
    parser.add_argument("--api_key", type=str, default="", help="Optional API key override")
    parser.add_argument("--text_only", action="store_true", help="Ignore video frames and generate summaries from translated script only")
    parser.add_argument(
        "--face_crop",
        action="store_true",
        help="Reuse FaceEncoder face-mesh alignment/crop logic before sending frames to the model.",
    )
    parser.add_argument(
        "--debug_face_dir",
        type=str,
        default="",
        help="Optional directory to save original and cropped face images for inspection.",
    )
    parser.add_argument("--num_frames", type=int, default=1, choices=[1, 3])
    parser.add_argument("--media_mode", type=str, default="base64", choices=["base64", "file"])
    parser.add_argument("--frame_cache_dir", type=str, default="out/meld_pseudo_frames")
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device_map", type=str, default="auto", help="Transformers backend device_map setting")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow custom model code for the transformers backend")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Keep model-specific reasoning/thinking mode enabled when supported.",
    )
    parser.add_argument("--allow_text_only", action="store_true")
    parser.add_argument("--use_emotion_label", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cache(cache_path: str) -> dict:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_cache(cache_path: str, cache: dict) -> None:
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, cache_path)


def parse_meld_video_path(videos_root: Path, sample_id: str) -> Path:
    match = MELD_ID_RE.match(sample_id)
    if not match:
        raise ValueError(f"Unexpected MELD sample id: {sample_id}")
    dialogue_id = int(match.group("dialogue"))
    utterance_id = int(match.group("utterance"))
    return videos_root / f"dia{dialogue_id}_utt{utterance_id}.mp4"


def time_str_to_seconds(value: str) -> float:
    if not isinstance(value, str):
        return 0.0
    s = value.strip()
    if not s:
        return 0.0
    hour_str, minute_str, second_ms_str = s.split(":")
    second_str, millisecond_str = second_ms_str.split(",")
    return (
        int(hour_str) * 3600
        + int(minute_str) * 60
        + int(second_str)
        + int(millisecond_str) / 1000.0
    )


def load_meld_time_ranges(csv_path: str) -> dict[str, tuple[float, float]]:
    if not csv_path:
        return {}
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {path}")

    mapping: dict[str, tuple[float, float]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                dialogue_id = int(str(row.get("Dialogue_ID", "")).strip())
                utterance_id = int(str(row.get("Utterance_ID", "")).strip())
            except ValueError:
                continue
            sample_id = f"meld_d{dialogue_id}_u{utterance_id}"
            start_t = time_str_to_seconds(str(row.get("StartTime", "")))
            end_t = time_str_to_seconds(str(row.get("EndTime", "")))
            mapping[sample_id] = (start_t, end_t)
    return mapping


def choose_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= 1 or num_frames == 1:
        return [max(0, frame_count // 2)]
    return [
        max(0, int(frame_count * 0.2)),
        max(0, int(frame_count * 0.5)),
        max(0, int(frame_count * 0.8)),
    ]


def extract_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = choose_frame_indices(frame_count, num_frames)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    capture.release()
    return frames


def extract_frames_at_times(video_path: Path, timestamps: list[float]) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    for t_sec in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    if not frames:
        capture.set(cv2.CAP_PROP_POS_MSEC, 0.0)
        ok, frame_bgr = capture.read()
        if ok and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    capture.release()
    return frames


def choose_timepoints(start_t: float, end_t: float, num_frames: int) -> list[float]:
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


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, float(max_side) / max(width, height))
    if scale == 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def crop_face_with_encoder(face_encoder, frame_bgr: np.ndarray) -> Image.Image | None:
    try:
        img_h, img_w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = face_encoder.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None

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
            return None
        if face_bgr.shape[0] < 10 or face_bgr.shape[1] < 10:
            return None

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)
    except Exception:
        return None


def maybe_crop_frames(face_encoder, frames: list[Image.Image]) -> list[Image.Image]:
    if face_encoder is None:
        return frames

    cropped_frames: list[Image.Image] = []
    for frame in frames:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        cropped = crop_face_with_encoder(face_encoder, frame_bgr)
        if cropped is not None:
            cropped_frames.append(cropped)
    return cropped_frames


def save_debug_face_images(
    debug_face_dir: Path | None,
    sample_id: str,
    frames: list[Image.Image],
    cropped_frames: list[Image.Image],
) -> None:
    if debug_face_dir is None:
        return

    debug_face_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(debug_face_dir / f"{sample_id}_f{index}_orig.jpg", format="JPEG", quality=95)

    if not cropped_frames:
        marker = debug_face_dir / f"{sample_id}_crop_fail.txt"
        marker.write_text("face crop failed", encoding="utf-8")
        return

    for index, cropped in enumerate(cropped_frames):
        cropped.save(debug_face_dir / f"{sample_id}_f{index}_crop.jpg", format="JPEG", quality=95)


def image_to_data_url(image: Image.Image, jpeg_quality: int) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def save_frame_to_cache(
    image: Image.Image,
    frame_cache_dir: Path,
    sample_id: str,
    frame_index: int,
    jpeg_quality: int,
) -> str:
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_cache_dir / f"{sample_id}_f{frame_index}.jpg"
    image.save(frame_path, format="JPEG", quality=jpeg_quality)
    # vLLM local media mode expects a filesystem path string in image_url.url.
    return str(frame_path.resolve())


def load_style_examples(style_examples_pt: str, limit: int) -> list[str]:
    path = Path(style_examples_pt)
    if not path.exists():
        return []
    dataset = torch.load(path, map_location="cpu")
    seen = set()
    examples = []
    for item in dataset:
        text = str(item.get("target_text", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        examples.append(text)
        if len(examples) >= limit:
            break
    return examples


def build_prompts(style_examples: Iterable[str], transcript: str, emotion: str | None) -> tuple[str, str]:
    style_block = "\n".join(f"- {example}" for example in style_examples) or "- 눈썹을 찌푸리고 낮은 목소리로 불만을 천천히 말함."

    system_prompt = (
        "You are a multimodal annotator that writes short Korean observational summaries for short dialogue clips. "
        "Describe only observable cues from facial expression, tone, and spoken content. "
        "Do not mention model uncertainty, camera, or analysis steps. "
        "Avoid explicit emotion labels such as 화남, 슬픔, 행복, angry, sad, happy. "
        "Write one Korean sentence in the style of an accessible situation description."
    )

    label_hint = ""
    if emotion:
        label_hint = (
            "\n[참고 라벨]\n"
            f"- 이 샘플의 감정 라벨: {emotion}\n"
            "- 라벨은 문장을 베끼는 용도가 아니라, 관찰 가능한 표현을 놓치지 않기 위한 참고용이다.\n"
        )

    user_prompt = (
        "[스타일 예시]\n"
        f"{style_block}\n\n"
        "[현재 대사]\n"
        f"- {transcript}\n"
        f"{label_hint}\n"
        "[작성 규칙]\n"
        "1. 한 문장으로 작성한다.\n"
        "2. 얼굴 표정, 말투, 발화 내용에서 보이는 단서만 쓴다.\n"
        "3. 추측성 심리 분석 대신 관찰 문장으로 쓴다.\n"
        "4. 감정 라벨 단어를 직접 쓰지 않는다.\n"
        "5. 한국어 데이터셋 요약문처럼 자연스럽고 간결하게 쓴다.\n"
    )

    return system_prompt, user_prompt


def build_multimodal_content(
    user_prompt: str,
    frames: list[Image.Image],
    sample_id: str,
    media_mode: str,
    frame_cache_dir: Path,
    max_side: int,
    jpeg_quality: int,
) -> list[dict]:
    content = [{"type": "text", "text": user_prompt}]
    for index, frame in enumerate(frames):
        resized = resize_image(frame, max_side=max_side)
        if media_mode == "file":
            image_url = save_frame_to_cache(
                resized,
                frame_cache_dir=frame_cache_dir,
                sample_id=sample_id,
                frame_index=index,
                jpeg_quality=jpeg_quality,
            )
        else:
            image_url = image_to_data_url(resized, jpeg_quality=jpeg_quality)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )
    return content


def request_summary(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    frames: list[Image.Image],
    sample_id: str,
    media_mode: str,
    frame_cache_dir: Path,
    max_side: int,
    jpeg_quality: int,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> str:
    content = build_multimodal_content(
        user_prompt=user_prompt,
        frames=frames,
        sample_id=sample_id,
        media_mode=media_mode,
        frame_cache_dir=frame_cache_dir,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
    )
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not enable_thinking:
        # Qwen 3.x can spend the whole completion budget on reasoning and leave
        # the visible assistant content empty unless thinking is disabled.
        request_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    response = client.chat.completions.create(
        **request_kwargs,
    )
    message = response.choices[0].message
    text = (message.content or "").strip()
    if text:
        return text

    reasoning = getattr(message, "reasoning", "")
    if reasoning:
        raise RuntimeError(
            "Model returned only reasoning text with empty content. "
            "Retry with thinking disabled or keep --enable_thinking off."
        )
    raise RuntimeError("Model returned an empty response.")


def resolve_torch_dtype(name: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_transformers_text_backend(model_id: str, device_map: str, torch_dtype: str, trust_remote_code: bool) -> dict:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:
        raise RuntimeError(
            "Transformers backend requires a newer Hugging Face transformers install. "
            "Qwen3.5's model card recommends installing the latest transformers."
        ) from error

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    resolved_dtype = resolve_torch_dtype(torch_dtype)
    if resolved_dtype != "auto":
        model_kwargs["torch_dtype"] = resolved_dtype
    if device_map:
        model_kwargs["device_map"] = device_map

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as error:
        raise RuntimeError(
            "Failed to load the local transformers backend. "
            "For Qwen3.5, use a recent transformers build as recommended by the official model card."
        ) from error

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return {"tokenizer": tokenizer, "model": model}


def load_transformers_multimodal_backend(model_id: str, device_map: str, torch_dtype: str, trust_remote_code: bool) -> dict:
    try:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    except Exception as error:
        raise RuntimeError(
            "Transformers multimodal backend requires a recent Hugging Face transformers install "
            "with Qwen3.5 multimodal support."
        ) from error

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    resolved_dtype = resolve_torch_dtype(torch_dtype)
    if resolved_dtype != "auto":
        model_kwargs["torch_dtype"] = resolved_dtype
    if device_map:
        model_kwargs["device_map"] = device_map

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    except Exception as error:
        raise RuntimeError(
            "Failed to load the local transformers multimodal backend. "
            "For Qwen3.5, use a recent transformers build as recommended by the official model card."
        ) from error

    return {"processor": processor, "model": model}


def build_tokenizer_chat_inputs(tokenizer, messages: list[dict], enable_thinking: bool) -> dict:
    apply_kwargs = {
        "add_generation_prompt": True,
    }
    if not enable_thinking:
        apply_kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_kwargs,
        )
    except TypeError:
        fallback_kwargs = {"add_generation_prompt": True}
        if not enable_thinking:
            fallback_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **fallback_kwargs,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                **fallback_kwargs,
            )
            return tokenizer(prompt_text, return_tensors="pt")


def strip_reasoning_markup(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("Thinking Process:"):
        text = text.split("\n\n", 1)[-1].strip()
    return text


def infer_model_device(model) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device

    hf_device_map = getattr(model, "hf_device_map", None) or {}
    for device_name in hf_device_map.values():
        if isinstance(device_name, str) and device_name not in {"cpu", "disk"}:
            return torch.device(device_name)
    return torch.device("cpu")


def request_summary_transformers(
    backend: dict,
    system_prompt: str,
    user_prompt: str,
    frames: list[Image.Image],
    max_side: int,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> str:
    model = backend["model"]

    if "processor" in backend:
        processor = backend["processor"]
        user_content: list[dict] = []
        for frame in frames:
            user_content.append(
                {
                    "type": "image",
                    # Inference from the multimodal chat template docs: decoded image
                    # objects can be passed directly, analogous to decoded video objects.
                    "image": resize_image(frame, max_side=max_side),
                }
            )
        user_content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        apply_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if not enable_thinking:
            apply_kwargs["enable_thinking"] = False
        try:
            model_inputs = processor.apply_chat_template(messages, **apply_kwargs)
        except TypeError:
            if "enable_thinking" in apply_kwargs:
                apply_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
                apply_kwargs.pop("enable_thinking", None)
            model_inputs = processor.apply_chat_template(messages, **apply_kwargs)

        tokenizer = getattr(processor, "tokenizer", None)
        model_device = infer_model_device(model)
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(model_device)
        else:
            model_inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        decode_fn = lambda ids: processor.batch_decode(  # noqa: E731
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    else:
        tokenizer = backend["tokenizer"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        model_inputs = build_tokenizer_chat_inputs(tokenizer, messages, enable_thinking=enable_thinking)
        model_device = infer_model_device(model)
        model_inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        decode_fn = lambda ids: tokenizer.batch_decode(  # noqa: E731
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, **generation_kwargs)

    input_length = model_inputs["input_ids"].shape[1]
    completion_ids = generated_ids[:, input_length:]
    text = decode_fn(completion_ids)
    text = strip_reasoning_markup(text)
    if not text:
        raise RuntimeError("Local transformers generation returned an empty response.")
    return text


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_pt)
    videos_root = Path(args.videos_root)
    output_path = Path(args.output_pt)
    cache_path = args.cache_json or str(output_path.with_suffix(".cache.json"))
    frame_cache_dir = Path(args.frame_cache_dir)
    debug_face_dir = Path(args.debug_face_dir) if args.debug_face_dir else None
    meld_time_ranges = load_meld_time_ranges(args.meld_csv)

    dataset = torch.load(input_path, map_location="cpu")
    style_examples = load_style_examples(args.style_examples_pt, args.style_examples)

    client = None
    local_backend = None
    face_encoder = None
    if args.backend == "openai":
        from openai import OpenAI

        api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        client_kwargs = {"api_key": api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        client = OpenAI(**client_kwargs)
    else:
        if args.text_only:
            local_backend = load_transformers_text_backend(
                model_id=args.model,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            local_backend = load_transformers_multimodal_backend(
                model_id=args.model,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
            )
    if args.face_crop and not args.text_only:
        from muton.encoders import FaceEncoder

        face_encoder = FaceEncoder()

    cache = load_cache(cache_path)
    output = []
    processed = 0
    attempted = 0

    for item in dataset:
        sample = dict(item)
        sample_id = str(sample.get("id", "")).strip()
        transcript = str(sample.get("script", "")).strip()
        if not sample_id or not transcript:
            output.append(sample)
            continue

        attempted += 1
        if args.limit and attempted > args.limit:
            break

        if not args.overwrite and sample_id in cache:
            sample["original_target_text"] = sample.get("target_text", "")
            sample["target_text"] = cache[sample_id]
            sample["pseudo_target_text"] = cache[sample_id]
            output.append(sample)
            continue

        frames = []
        video_path = parse_meld_video_path(videos_root, sample_id)
        if not args.text_only:
            if not video_path.exists():
                print(f"[skip] missing video: {sample_id} -> {video_path}")
                output.append(sample)
                continue
            if sample_id in meld_time_ranges:
                start_t, end_t = meld_time_ranges[sample_id]
                frames = extract_frames_at_times(video_path, choose_timepoints(start_t, end_t, args.num_frames))
            else:
                frames = extract_frames(video_path, args.num_frames)

        if not args.text_only and not frames and not args.allow_text_only:
            print(f"[skip] frame read fail: {sample_id} -> {video_path}")
            output.append(sample)
            continue

        if not args.text_only and args.face_crop:
            original_frames = frames
            frames = maybe_crop_frames(face_encoder, frames)
            save_debug_face_images(debug_face_dir, sample_id, original_frames, frames)
            if not frames and not args.allow_text_only:
                print(f"[skip] face crop fail: {sample_id} -> {video_path}")
                output.append(sample)
                continue

        emotion = str(sample.get("emotion", "")).strip() if args.use_emotion_label else None
        system_prompt, user_prompt = build_prompts(style_examples, transcript, emotion)
        try:
            if args.backend == "openai":
                pseudo = request_summary(
                    client=client,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    frames=frames,
                    sample_id=sample_id,
                    media_mode=args.media_mode,
                    frame_cache_dir=frame_cache_dir,
                    max_side=args.max_side,
                    jpeg_quality=args.jpeg_quality,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                )
            else:
                pseudo = request_summary_transformers(
                    backend=local_backend,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    frames=frames,
                    max_side=args.max_side,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                )
        except Exception as error:
            print(f"[error] {sample_id}: {error}")
            output.append(sample)
            continue

        cache[sample_id] = pseudo
        sample["original_target_text"] = sample.get("target_text", "")
        sample["target_text"] = pseudo
        sample["pseudo_target_text"] = pseudo
        sample["pseudo_summary_model"] = args.model
        sample["pseudo_summary_has_image"] = bool(frames)
        sample["pseudo_summary_face_crop"] = bool(args.face_crop and frames)
        output.append(sample)

        processed += 1
        if processed % 20 == 0:
            print(f"[ok] generated {processed} summaries")
            save_cache(cache_path, cache)

    save_cache(cache_path, cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"saved: {output_path}")
    print(f"generated: {processed}")
    print(f"attempted: {attempted}")


if __name__ == "__main__":
    main()
