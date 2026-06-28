# embedding.py

import json
import os

import cv2
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from muton.config import env_path
from muton.encoders import AudioEncoder, FaceEncoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACE_ROOT = env_path("MUTON_FACE_ROOT", "data/face_crops")
AUDIO_ROOT = env_path("MUTON_AUDIO_ROOT", "data/audio")
JSON_PATH = env_path("MUTON_MULTI_TEXT_JSON", "preprocessing/multi_text.json")
OUT_PATH = env_path("MUTON_FUSION_DATASET", "data/fusion_dataset.pt")

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
text_model = AutoModel.from_pretrained("klue/roberta-small").to(DEVICE).eval()
face_encoder = FaceEncoder()
audio_encoder = AudioEncoder()
print("Models loaded")


def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def get_summary_text(item):
    for key in ["summary_text", "summary_textc", "accessible_emotion_desc"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def encode_text(script: str):
    inputs = tokenizer(
        script,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = text_model(**inputs, return_dict=True)
        return mean_pool(outputs.last_hidden_state, inputs["attention_mask"]).squeeze(0).cpu()


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    dataset = []
    print("Start embedding extraction")

    for clip_key, items in tqdm(data.items()):
        if not clip_key.startswith("clip_"):
            continue

        pid = clip_key.replace("clip_", "").strip()
        face_dir = FACE_ROOT / pid
        audio_dir = AUDIO_ROOT / pid
        if not face_dir.is_dir() or not audio_dir.is_dir():
            continue

        for item in items:
            name = item.get("name", "").strip()
            if not name:
                continue

            face_path = face_dir / f"{name}.jpg"
            if not face_path.exists():
                face_path = face_dir / f"{name}.jpeg"
            audio_path = audio_dir / f"{name}.wav"

            if not face_path.exists() or not audio_path.exists():
                continue

            image = cv2.imread(str(face_path))
            if image is None:
                continue

            face_out = face_encoder.encode_frame(image)
            if face_out is None:
                continue

            face_vec = torch.tensor(face_out["face_vec"], dtype=torch.float32)
            face_emo_logits = torch.tensor(face_out["face_emotion_logits"], dtype=torch.float32)
            if face_vec.shape != (768,) or face_emo_logits.shape != (7,):
                continue

            pcm, _ = sf.read(str(audio_path))
            if getattr(pcm, "ndim", 1) > 1:
                pcm = pcm.mean(axis=1)

            feats = audio_encoder.extract_features_from_pcm(np.asarray(pcm, dtype=np.float32))
            audio_content = torch.tensor(feats["content"], dtype=torch.float32)
            audio_speaker = torch.tensor(feats["speaker"], dtype=torch.float32)
            audio_prosody = torch.tensor(feats["prosody"], dtype=torch.float32)
            if audio_content.shape != (768,):
                continue

            script = item.get("script", "")
            summary = get_summary_text(item)
            if not summary:
                continue

            dataset.append(
                {
                    "id": name,
                    "person_id": pid,
                    "face_vec": face_vec,
                    "face_emo_logits": face_emo_logits,
                    "audio_content": audio_content,
                    "audio_speaker": audio_speaker,
                    "audio_prosody": audio_prosody,
                    "text": encode_text(script),
                    "emotion": item.get("emotion", "Neutral"),
                    "arousal": float(item.get("arousal", 5.0)),
                    "valence": float(item.get("valence", 5.0)),
                    "target_text": summary,
                    "script": script,
                }
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, str(OUT_PATH))
    print(f"Saved dataset -> {OUT_PATH}")
    print(f"Total samples: {len(dataset)}")


if __name__ == "__main__":
    main()
