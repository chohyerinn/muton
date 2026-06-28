# -*- coding: utf-8 -*-
"""
train_fusion_ko_final.py
- Load a MELD pretrain checkpoint
- Replace the emotion head for the Korean label space
- Fine-tune on the full Korean fusion dataset
"""

import argparse
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_fusion_meld import FusionTransformer


KO_EMO2ID = {
    "angry": 0,
    "dislike": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_f32(batch, key):
    return torch.from_numpy(
        np.stack([np.asarray(item[key], dtype=np.float32) for item in batch], axis=0)
    )


def ko_collate_fn(batch):
    return {
        "face_vec": _to_f32(batch, "face_vec"),
        "face_emo": _to_f32(batch, "face_emo_logits"),
        "a_cont": _to_f32(batch, "audio_content"),
        "a_spk": _to_f32(batch, "audio_speaker"),
        "a_pros": _to_f32(batch, "audio_prosody"),
        "text": _to_f32(batch, "text"),
        "emo": torch.tensor(
            [KO_EMO2ID.get(str(item["emotion"]).strip().lower(), KO_EMO2ID["neutral"]) for item in batch],
            dtype=torch.long,
        ),
        "arousal": torch.tensor([float(item.get("arousal", 5.0)) for item in batch], dtype=torch.float32),
        "valence": torch.tensor([float(item.get("valence", 5.0)) for item in batch], dtype=torch.float32),
    }


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def freeze_backbone(model: FusionTransformer, unfreeze_last_nlayers: int = 0) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "head_" in name:
            param.requires_grad = True

    if unfreeze_last_nlayers <= 0:
        return

    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "encoder"):
        layers = model.encoder.layers
    if layers is None:
        raise AttributeError("FusionTransformer does not expose transformer layers.")

    for layer_index in range(max(0, len(layers) - unfreeze_last_nlayers), len(layers)):
        for param in layers[layer_index].parameters():
            param.requires_grad = True


@torch.no_grad()
def eval_trainset(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    mse_a = 0.0
    mse_v = 0.0

    for batch in loader:
        for key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

        emo_logits, a_pred, v_pred = model(batch)
        pred = emo_logits.argmax(dim=-1)
        target = batch["emo"]

        correct += (pred == target).sum().item()
        total += target.numel()
        mse_a += F.mse_loss(a_pred, batch["arousal"], reduction="sum").item()
        mse_v += F.mse_loss(v_pred, batch["valence"], reduction="sum").item()

    return correct / max(1, total), mse_a / max(1, total), mse_v / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_ckpt", type=str, required=True, help="MELD pretrain best.pt")
    parser.add_argument("--ko_pt", type=str, required=True, help="Korean fusion_dataset.pt")
    parser.add_argument("--out_pt", type=str, default="out/fusion_ko_final/final.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze_last_nlayers", type=int, default=0)
    parser.add_argument("--w_emo", type=float, default=1.0)
    parser.add_argument("--w_arousal", type=float, default=0.5)
    parser.add_argument("--w_valence", type=float, default=0.5)
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save an intermediate checkpoint every n epochs when > 0.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(args.ko_pt)
    print("KO N =", len(data))
    print("KO label dist:", Counter([str(item["emotion"]).strip().lower() for item in data]))

    loader = DataLoader(
        ListDataset(data),
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=ko_collate_fn,
    )

    pkg = torch.load(args.pre_ckpt, map_location="cpu")
    cfg = pkg.get("args", {})
    backbone_cfg = {
        "d_model": cfg.get("d_model", 256),
        "nhead": cfg.get("nhead", 8),
        "nlayers": cfg.get("nlayers", 4),
    }

    model = FusionTransformer(
        d_model=backbone_cfg["d_model"],
        nhead=backbone_cfg["nhead"],
        nlayers=backbone_cfg["nlayers"],
        num_emotions=7,
    ).to(device)
    model.load_state_dict(pkg["model"], strict=True)

    model.head_emo = nn.Linear(model.head_emo.in_features, len(KO_EMO2ID)).to(device)
    freeze_backbone(model, unfreeze_last_nlayers=args.unfreeze_last_nlayers)

    trainable = [param for param in model.parameters() if param.requires_grad]
    print("trainable params:", sum(param.numel() for param in trainable))

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in loader:
            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)

            emo_logits, a_pred, v_pred = model(batch)
            loss = (
                args.w_emo * F.cross_entropy(emo_logits, batch["emo"])
                + args.w_arousal * F.mse_loss(a_pred, batch["arousal"])
                + args.w_valence * F.mse_loss(v_pred, batch["valence"])
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        acc, mse_a, mse_v = eval_trainset(model, loader, device)

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[epoch {epoch}] loss={avg_loss:.4f} "
                f"train_acc={acc:.4f} mse_a={mse_a:.4f} mse_v={mse_v:.4f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if args.save_every and epoch % args.save_every == 0:
            mid_path = args.out_pt.replace(".pt", f"_e{epoch}.pt")
            torch.save(
                {
                    "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "pre_ckpt": args.pre_ckpt,
                    "ko_emo2id": KO_EMO2ID,
                    "args": vars(args),
                    "backbone_cfg": backbone_cfg,
                },
                mid_path,
            )
            print("saved mid:", mid_path)

    torch.save(
        {
            "model": best_state if best_state is not None else model.state_dict(),
            "pre_ckpt": args.pre_ckpt,
            "ko_emo2id": KO_EMO2ID,
            "args": vars(args),
            "backbone_cfg": backbone_cfg,
        },
        args.out_pt,
    )
    print("SAVED:", args.out_pt)
    print("BEST train_loss:", best_loss)


if __name__ == "__main__":
    main()
