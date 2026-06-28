# -*- coding: utf-8 -*-
"""
eval_ko_loocv.py
- LOOCV for Korean fusion_dataset.pt (N=36)
- Each fold:
  - load MELD pretrain ckpt
  - replace head_emo to 6-class
  - freeze backbone (optionally unfreeze last N encoder layers)
  - train on N-1
  - evaluate on 1

Outputs:
- overall acc
- confusion matrix
- per-class precision/recall/F1
- mse(arousal), mse(valence)
"""

import os
import argparse
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_fusion_meld import FusionTransformer

KO_EMO2ID = {
    "angry": 0,
    "dislike": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5,
}
ID2KO = {v: k for k, v in KO_EMO2ID.items()}

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _to_f32(batch, key):
    return torch.from_numpy(np.stack([np.asarray(b[key], dtype=np.float32) for b in batch], axis=0))

def ko_collate_fn(batch):
    face_vec = _to_f32(batch, "face_vec")
    face_emo = _to_f32(batch, "face_emo_logits")
    a_cont  = _to_f32(batch, "audio_content")
    a_spk   = _to_f32(batch, "audio_speaker")
    a_pros  = _to_f32(batch, "audio_prosody")
    text    = _to_f32(batch, "text")

    emo = torch.tensor(
        [KO_EMO2ID.get(str(b["emotion"]).strip().lower(), KO_EMO2ID["neutral"]) for b in batch],
        dtype=torch.long
    )
    arousal = torch.tensor([float(b.get("arousal", 5.0)) for b in batch], dtype=torch.float32)
    valence = torch.tensor([float(b.get("valence", 5.0)) for b in batch], dtype=torch.float32)

    return {
        "face_vec": face_vec,
        "face_emo": face_emo,
        "a_cont": a_cont,
        "a_spk": a_spk,
        "a_pros": a_pros,
        "text": text,
        "emo": emo,
        "arousal": arousal,
        "valence": valence,
    }

class ListDataset(Dataset):
    def __init__(self, data_list): self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def freeze_backbone(model: FusionTransformer, unfreeze_last_nlayers: int = 0):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "head_" in name:
            p.requires_grad = True
    if unfreeze_last_nlayers > 0:
        layers = getattr(model, "layers", None)
        if layers is None and hasattr(model, "encoder"):
            layers = model.encoder.layers
        if layers is None:
            raise AttributeError("FusionTransformer does not expose transformer layers.")
        n = len(layers)
        for li in range(max(0, n - unfreeze_last_nlayers), n):
            for p in layers[li].parameters():
                p.requires_grad = True

@torch.no_grad()
def eval_one(model, batch, device):
    model.eval()
    for k in batch:
        batch[k] = batch[k].to(device, non_blocking=True)
    emo_logits, a_pred, v_pred = model(batch)
    y = batch["emo"]
    p = emo_logits.argmax(dim=-1)
    acc = float((p == y).item())  # single sample
    mse_a = float(F.mse_loss(a_pred, batch["arousal"]).item())
    mse_v = float(F.mse_loss(v_pred, batch["valence"]).item())
    return int(y.item()), int(p.item()), acc, mse_a, mse_v

def confusion_matrix(y_true, y_pred, n_cls):
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def per_class_stats(cm):
    stats = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        stats.append((prec, rec, f1))
    return stats

def build_model_from_pretrain(pre_ckpt_path: str, device: str):
    pkg = torch.load(pre_ckpt_path, map_location="cpu")
    cfg = pkg.get("args", {})
    d_model = cfg.get("d_model", 256)
    nhead = cfg.get("nhead", 8)
    nlayers = cfg.get("nlayers", 4)

    model = FusionTransformer(d_model=d_model, nhead=nhead, nlayers=nlayers, num_emotions=7).to(device)
    model.load_state_dict(pkg["model"], strict=True)

    # replace to 6-class
    d_in = model.head_emo.in_features
    model.head_emo = nn.Linear(d_in, 6).to(device)
    return model, {"d_model": d_model, "nhead": nhead, "nlayers": nlayers}

def train_on_split(model, train_loader, device, epochs, lr, w_emo, w_a, w_v):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            emo_logits, a_pred, v_pred = model(batch)
            loss_emo = F.cross_entropy(emo_logits, batch["emo"])
            loss_a = F.mse_loss(a_pred, batch["arousal"])
            loss_v = F.mse_loss(v_pred, batch["valence"])
            loss = w_emo * loss_emo + w_a * loss_a + w_v * loss_v

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_ckpt", required=True)
    ap.add_argument("--ko_pt", required=True)
    ap.add_argument("--out_dir", default="out/ko_loocv")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unfreeze_last_nlayers", type=int, default=0)
    ap.add_argument("--w_emo", type=float, default=1.0)
    ap.add_argument("--w_arousal", type=float, default=0.5)
    ap.add_argument("--w_valence", type=float, default=0.5)
    ap.add_argument("--save_fold_ckpt", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(args.ko_pt)
    N = len(data)
    print("KO N =", N)
    print("KO label dist:", Counter([str(x["emotion"]).strip().lower() for x in data]))

    y_true_all, y_pred_all = [], []
    accs, mses_a, mses_v = [], [], []

    for test_idx in range(N):
        # split
        test_item = data[test_idx]
        train_items = [data[i] for i in range(N) if i != test_idx]

        # model
        model, backbone_cfg = build_model_from_pretrain(args.pre_ckpt, device)
        freeze_backbone(model, unfreeze_last_nlayers=args.unfreeze_last_nlayers)

        # loader
        train_loader = DataLoader(
            ListDataset(train_items),
            batch_size=args.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=ko_collate_fn
        )

        # train
        train_on_split(
            model, train_loader, device,
            epochs=args.epochs,
            lr=args.lr,
            w_emo=args.w_emo,
            w_a=args.w_arousal,
            w_v=args.w_valence
        )

        # eval on single
        test_batch = ko_collate_fn([test_item])
        yt, yp, acc, mse_a, mse_v = eval_one(model, test_batch, device)

        y_true_all.append(yt)
        y_pred_all.append(yp)
        accs.append(acc)
        mses_a.append(mse_a)
        mses_v.append(mse_v)

        if (test_idx + 1) % 5 == 0 or test_idx == 0:
            print(f"[fold {test_idx+1:02d}/{N}] acc={np.mean(accs):.3f} mse_a={np.mean(mses_a):.3f} mse_v={np.mean(mses_v):.3f}")

        if args.save_fold_ckpt:
            torch.save(
                {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "test_idx": test_idx,
                    "backbone_cfg": backbone_cfg,
                    "args": vars(args),
                },
                os.path.join(args.out_dir, f"fold{test_idx:02d}.pt")
            )

    # summary
    mean_acc = float(np.mean(accs))
    mean_mse_a = float(np.mean(mses_a))
    mean_mse_v = float(np.mean(mses_v))

    print("\n==== LOOCV SUMMARY ====")
    print(f"acc={mean_acc:.4f}  mse_a={mean_mse_a:.4f}  mse_v={mean_mse_v:.4f}")

    cm = confusion_matrix(y_true_all, y_pred_all, 6)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    stats = per_class_stats(cm)
    print("\nPer-class (Precision, Recall, F1):")
    for i in range(6):
        p, r, f1 = stats[i]
        sup = int(cm[i, :].sum())
        print(f"{i:2d} {ID2KO[i]:10s} P={p:.3f} R={r:.3f} F1={f1:.3f} support={sup}")

    # save report
    out_path = os.path.join(args.out_dir, "loocv_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("==== LOOCV SUMMARY ====\n")
        f.write(f"acc={mean_acc:.4f}  mse_a={mean_mse_a:.4f}  mse_v={mean_mse_v:.4f}\n\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("Per-class (Precision, Recall, F1):\n")
        for i in range(6):
            p, r, f1 = stats[i]
            sup = int(cm[i, :].sum())
            f.write(f"{i:2d} {ID2KO[i]:10s} P={p:.3f} R={r:.3f} F1={f1:.3f} support={sup}\n")

    print("\nSaved report:", out_path)

if __name__ == "__main__":
    main()
