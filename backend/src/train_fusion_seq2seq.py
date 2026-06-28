# -*- coding: utf-8 -*-
"""
train_fusion_seq2seq.py

Experimental path:
- reuse the multimodal fusion encoder
- predict emotion/arousal/valence as before
- generate a Korean situation summary by letting a seq2seq decoder
  cross-attend directly to fusion hidden states
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

try:
    from src.fusion_seq2seq import FusionEncoderDecoder, load_compatible_fusion_weights
except ImportError:
    from fusion_seq2seq import FusionEncoderDecoder, load_compatible_fusion_weights


KO_EMO2ID = {
    "angry": 0,
    "dislike": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5,
}

MELD_EMO2ID = {
    "angry": 0,
    "sad": 1,
    "disgust": 2,
    "surprise": 3,
    "happy": 4,
    "neutral": 5,
    "fear": 6,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_f32(batch: List[dict], key: str) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([np.asarray(item[key], dtype=np.float32) for item in batch], axis=0)
    )


class ListDataset(Dataset):
    def __init__(self, data_list: List[dict]):
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return self.data[index]


def normalize_label(label: str) -> str:
    return str(label or "").strip().lower()


def infer_emotion_space(data: List[dict]) -> str:
    labels = {normalize_label(item.get("emotion", "")) for item in data}
    if "dislike" in labels:
        return "ko"
    if "fear" in labels or "disgust" in labels:
        return "meld"
    return "ko"


def resolve_emotion_to_id(emotion_space: str, data: List[dict]) -> Dict[str, int]:
    if emotion_space == "auto":
        emotion_space = infer_emotion_space(data)

    if emotion_space == "ko":
        return KO_EMO2ID
    if emotion_space == "meld":
        return MELD_EMO2ID

    raise ValueError(f"Unsupported emotion_space: {emotion_space}")


def build_collate_fn(tokenizer: AutoTokenizer, max_target_len: int, emotion_to_id: Dict[str, int]):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("The decoder tokenizer must define a pad token.")

    def collate_fn(batch: List[dict]) -> Dict[str, torch.Tensor | List[str]]:
        target_texts = [str(item.get("target_text", "")).strip() for item in batch]
        label_tokens = tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_target_len,
        )
        labels = label_tokens["input_ids"]
        labels = labels.masked_fill(labels == pad_token_id, -100)

        return {
            "face_vec": _to_f32(batch, "face_vec"),
            "face_emo": _to_f32(batch, "face_emo_logits"),
            "a_cont": _to_f32(batch, "audio_content"),
            "a_spk": _to_f32(batch, "audio_speaker"),
            "a_pros": _to_f32(batch, "audio_prosody"),
            "text": _to_f32(batch, "text"),
            "emo": torch.tensor(
                [emotion_to_id.get(normalize_label(item.get("emotion", "")), emotion_to_id.get("neutral", 0)) for item in batch],
                dtype=torch.long,
            ),
            "arousal": torch.tensor([float(item.get("arousal", 5.0)) for item in batch], dtype=torch.float32),
            "valence": torch.tensor([float(item.get("valence", 5.0)) for item in batch], dtype=torch.float32),
            "labels": labels,
            "target_texts": target_texts,
            "scripts": [str(item.get("script", "")) for item in batch],
            "ids": [str(item.get("id", "")) for item in batch],
        }

    return collate_fn


def move_batch_to_device(batch: Dict[str, torch.Tensor | List[str]], device: str) -> Dict[str, torch.Tensor | List[str]]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def maybe_freeze_fusion_backbone(model: FusionEncoderDecoder, unfreeze_last_nlayers: int) -> None:
    for name, param in model.named_parameters():
        if name.startswith("decoder.") or name.startswith("memory_") or name.startswith("head_"):
            continue
        param.requires_grad = False

    if unfreeze_last_nlayers > 0:
        model.unfreeze_encoder_last_nlayers(unfreeze_last_nlayers)


def build_optimizer(model: FusionEncoderDecoder, lr_fusion: float, lr_decoder: float) -> torch.optim.Optimizer:
    fusion_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("decoder."):
            decoder_params.append(param)
        else:
            fusion_params.append(param)

    param_groups = []
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": lr_fusion})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": lr_decoder})

    return torch.optim.AdamW(param_groups, weight_decay=0.01)


@torch.no_grad()
def evaluate(
    model: FusionEncoderDecoder,
    loader: DataLoader,
    device: str,
    w_emo: float,
    w_arousal: float,
    w_valence: float,
    w_gen: float,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    total_gen = 0.0
    total_a = 0.0
    total_v = 0.0
    total_cls = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch, labels=batch["labels"])

        loss_cls = F.cross_entropy(outputs["emo_logits"], batch["emo"])
        loss_a = F.mse_loss(outputs["arousal"], batch["arousal"])
        loss_v = F.mse_loss(outputs["valence"], batch["valence"])
        loss_gen = outputs["gen_loss"] if outputs["gen_loss"] is not None else torch.tensor(0.0, device=device)
        loss = w_emo * loss_cls + w_arousal * loss_a + w_valence * loss_v + w_gen * loss_gen

        pred = outputs["emo_logits"].argmax(dim=-1)
        correct += (pred == batch["emo"]).sum().item()
        total += batch["emo"].numel()

        batch_size = batch["emo"].shape[0]
        total_loss += loss.item() * batch_size
        total_gen += loss_gen.item() * batch_size
        total_a += loss_a.item() * batch_size
        total_v += loss_v.item() * batch_size
        total_cls += loss_cls.item() * batch_size

    denom = max(1, total)
    return {
        "loss": total_loss / denom,
        "acc": correct / denom,
        "loss_cls": total_cls / denom,
        "loss_gen": total_gen / denom,
        "loss_a": total_a / denom,
        "loss_v": total_v / denom,
    }


@torch.no_grad()
def preview_generations(
    model: FusionEncoderDecoder,
    loader: DataLoader,
    tokenizer: AutoTokenizer,
    device: str,
    num_beams: int,
    max_new_tokens: int,
    preview_samples: int,
) -> List[str]:
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return []

    batch = move_batch_to_device(batch, device)
    generated_ids = model.generate(
        batch,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    previews = []
    for idx in range(min(preview_samples, len(generated_texts))):
        previews.append(
            (
                f"[{batch['ids'][idx]}] script={batch['scripts'][idx]!r}\n"
                f"  pred={generated_texts[idx].strip()}\n"
                f"  gold={batch['target_texts'][idx]!r}"
            )
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_ckpt", type=str, required=True, help="Fusion backbone checkpoint to warm-start from")
    parser.add_argument("--train_pt", type=str, default="", help="Training dataset (.pt)")
    parser.add_argument("--val_pt", type=str, default="", help="Validation dataset (.pt). If omitted, split train_pt")
    parser.add_argument("--ko_pt", type=str, default="", help="Deprecated alias for --train_pt")
    parser.add_argument("--out_pt", type=str, default="out/fusion_seq2seq/best.pt")
    parser.add_argument("--decoder_model", type=str, default="google/mt5-small")
    parser.add_argument("--emotion_space", type=str, default="auto", choices=["auto", "ko", "meld"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_target_len", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--lr_fusion", type=float, default=5e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--w_emo", type=float, default=1.0)
    parser.add_argument("--w_arousal", type=float, default=0.5)
    parser.add_argument("--w_valence", type=float, default=0.5)
    parser.add_argument("--w_gen", type=float, default=1.0)
    parser.add_argument("--freeze_fusion_backbone", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--unfreeze_last_nlayers", type=int, default=1)
    parser.add_argument("--preview_samples", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_pt = args.train_pt or args.ko_pt
    if not train_pt:
        raise RuntimeError("Provide --train_pt (or legacy --ko_pt).")

    train_data = torch.load(train_pt)
    train_data = [item for item in train_data if str(item.get("target_text", "")).strip()]
    if not train_data:
        raise RuntimeError("No samples with target_text were found in the dataset.")

    emotion_to_id = resolve_emotion_to_id(args.emotion_space, train_data)
    print("train N =", len(train_data))
    print("train label dist:", Counter([normalize_label(item.get("emotion", "")) for item in train_data]))
    print("emotion_to_id:", emotion_to_id)

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    collate_fn = build_collate_fn(tokenizer, args.max_target_len, emotion_to_id)

    if args.val_pt:
        val_data = torch.load(args.val_pt)
        val_data = [item for item in val_data if str(item.get("target_text", "")).strip()]
        train_ds = ListDataset(train_data)
        val_ds = ListDataset(val_data)
    else:
        full_dataset = ListDataset(train_data)
        val_size = int(len(full_dataset) * args.val_ratio)
        if val_size <= 0:
            val_size = min(1, len(full_dataset) - 1)
        train_size = len(full_dataset) - val_size
        if train_size <= 0:
            raise RuntimeError("Dataset too small. Increase data size or reduce val_ratio.")

        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    print("effective train size:", len(train_ds))
    print("effective val size:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    pkg = torch.load(args.pre_ckpt, map_location="cpu")
    cfg = pkg.get("backbone_cfg") or pkg.get("args", {})
    backbone_cfg = {
        "d_model": cfg.get("d_model", 256),
        "nhead": cfg.get("nhead", 8),
        "nlayers": cfg.get("nlayers", 4),
    }

    model = FusionEncoderDecoder(
        d_model=backbone_cfg["d_model"],
        nhead=backbone_cfg["nhead"],
        nlayers=backbone_cfg["nlayers"],
        num_emotions=len(emotion_to_id),
        decoder_model_name=args.decoder_model,
    ).to(device)

    ckpt_state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
    missing, skipped = load_compatible_fusion_weights(model, ckpt_state)
    print("warm-start loaded")
    print("missing keys:", len(missing))
    print("skipped/incompatible keys:", len(skipped))

    if args.freeze_fusion_backbone:
        maybe_freeze_fusion_backbone(model, args.unfreeze_last_nlayers)
    if args.freeze_decoder:
        model.freeze_decoder()

    trainable = [param for param in model.parameters() if param.requires_grad]
    print("trainable params:", sum(param.numel() for param in trainable))

    optimizer = build_optimizer(model, args.lr_fusion, args.lr_decoder)
    best_metric = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, labels=batch["labels"])

            loss_cls = F.cross_entropy(outputs["emo_logits"], batch["emo"])
            loss_a = F.mse_loss(outputs["arousal"], batch["arousal"])
            loss_v = F.mse_loss(outputs["valence"], batch["valence"])
            loss_gen = outputs["gen_loss"] if outputs["gen_loss"] is not None else torch.tensor(0.0, device=device)
            loss = (
                args.w_emo * loss_cls
                + args.w_arousal * loss_a
                + args.w_valence * loss_v
                + args.w_gen * loss_gen
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        train_loss = total_loss / max(1, steps)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.w_emo,
            args.w_arousal,
            args.w_valence,
            args.w_gen,
        )

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_gen={val_metrics['loss_gen']:.4f}"
        )

        if epoch == 1 or epoch % 5 == 0:
            for preview in preview_generations(
                model,
                val_loader,
                tokenizer,
                device,
                args.num_beams,
                args.max_new_tokens,
                args.preview_samples,
            ):
                print(preview)

        if val_metrics["loss"] < best_metric:
            best_metric = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "pre_ckpt": args.pre_ckpt,
                    "decoder_model": args.decoder_model,
                    "emotion_to_id": emotion_to_id,
                    "args": vars(args),
                    "backbone_cfg": backbone_cfg,
                },
                args.out_pt,
            )
            print("saved best:", args.out_pt)

    if best_state is None:
        raise RuntimeError("Training finished without saving a checkpoint.")


if __name__ == "__main__":
    main()
