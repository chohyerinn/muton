# -*- coding: utf-8 -*-
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

EMO2ID = {
    "Angry": 0,
    "Sad": 1,
    "Disgust": 2,
    "Surprise": 3,
    "Happy": 4,
    "Neutral": 5,
    "Fear": 6,
}

def _to_f32(batch, key):
    # list of (np.ndarray or list) -> np.ndarray -> torch tensor
    return torch.from_numpy(np.stack([np.asarray(b[key], dtype=np.float32) for b in batch], axis=0))

def collate_fn(batch):
    face_vec = _to_f32(batch, "face_vec")          # (B,768)
    face_emo = _to_f32(batch, "face_emo_logits")   # (B,7)
    a_cont  = _to_f32(batch, "audio_content")      # (B,768)
    a_spk   = _to_f32(batch, "audio_speaker")      # (B,768)
    a_pros  = _to_f32(batch, "audio_prosody")      # (B,768)
    text    = _to_f32(batch, "text")               # (B,768)

    emo = torch.tensor([EMO2ID.get(b["emotion"], EMO2ID["Neutral"]) for b in batch], dtype=torch.long)
    arousal = torch.tensor([float(b["arousal"]) for b in batch], dtype=torch.float32)
    valence = torch.tensor([float(b["valence"]) for b in batch], dtype=torch.float32)

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

class PTDictDataset(Dataset):
    def __init__(self, pt_path: str):
        self.data = torch.load(pt_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, nlayers=4, dropout=0.1, num_emotions=7):
        super().__init__()

        self.proj_face = nn.Linear(768, d_model)
        self.proj_a_cont = nn.Linear(768, d_model)
        self.proj_a_spk = nn.Linear(768, d_model)
        self.proj_a_pros = nn.Linear(768, d_model)
        self.proj_text = nn.Linear(768, d_model)
        self.proj_faceemo = nn.Linear(7, d_model)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        # 🔥 custom encoder layer (attn 반환 가능)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            ) for _ in range(nlayers)
        ])

        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nlayers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nlayers)])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            ) for _ in range(nlayers)
        ])

        self.head_emo = nn.Linear(d_model, num_emotions)
        self.head_arousal = nn.Linear(d_model, 1)
        self.head_valence = nn.Linear(d_model, 1)

    def forward(self, batch, return_attn=False):
        B = batch["face_vec"].shape[0]

        t0 = self.proj_face(batch["face_vec"])
        t1 = self.proj_faceemo(batch["face_emo"])
        t2 = self.proj_a_cont(batch["a_cont"])
        t3 = self.proj_a_spk(batch["a_spk"])
        t4 = self.proj_a_pros(batch["a_pros"])
        t5 = self.proj_text(batch["text"])

        tokens = torch.stack([t0, t1, t2, t3, t4, t5], dim=1)

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)

        attn_last = None

        for attn, norm1, norm2, ffn in zip(self.layers, self.norms1, self.norms2, self.ffns):
            attn_out, attn_weights = attn(x, x, x, need_weights=True, average_attn_weights=False)
            x = norm1(x + attn_out)
            x = norm2(x + ffn(x))
            attn_last = attn_weights   # 마지막 레이어 attn

        cls_vec = x[:, 0]

        emo_logits = self.head_emo(cls_vec)
        arousal = self.head_arousal(cls_vec).squeeze(-1)
        valence = self.head_valence(cls_vec).squeeze(-1)

        if return_attn:
            # CLS가 각 토큰에 준 attention (6개 modality)
            # attn shape: (B, heads, 7, 7)
            cls_attn = attn_last.mean(1)[:, 0, 1:]  # (B,6)
            return emo_logits, arousal, valence, cls_attn

        return emo_logits, arousal, valence


def run_eval(model, loader, device):
    model.eval()
    n = 0
    correct = 0
    mse_a = 0.0
    mse_v = 0.0
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            emo_logits, a_pred, v_pred = model(batch)
            emo = batch["emo"]
            ar = batch["arousal"]
            va = batch["valence"]

            pred = emo_logits.argmax(dim=-1)
            correct += (pred == emo).sum().item()
            n += emo.numel()

            mse_a += F.mse_loss(a_pred, ar, reduction="sum").item()
            mse_v += F.mse_loss(v_pred, va, reduction="sum").item()

    acc = correct / max(1, n)
    mse_a = mse_a / max(1, n)
    mse_v = mse_v / max(1, n)
    return acc, mse_a, mse_v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pt", type=str, required=True)
    ap.add_argument("--dev_pt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out/fusion_ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--w_emo", type=float, default=1.0)
    ap.add_argument("--w_arousal", type=float, default=0.5)
    ap.add_argument("--w_valence", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = PTDictDataset(args.train_pt)
    dev_ds = PTDictDataset(args.dev_pt)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = FusionTransformer(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            emo_logits, a_pred, v_pred = model(batch)

            loss_emo = F.cross_entropy(emo_logits, batch["emo"])
            loss_a = F.mse_loss(a_pred, batch["arousal"])
            loss_v = F.mse_loss(v_pred, batch["valence"])

            loss = args.w_emo * loss_emo + args.w_arousal * loss_a + args.w_valence * loss_v

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            steps += 1

        acc, mse_a, mse_v = run_eval(model, dev_loader, device)
        print(f"[epoch {epoch}] train_loss={total_loss/max(1,steps):.4f} dev_acc={acc:.4f} dev_mse_a={mse_a:.4f} dev_mse_v={mse_v:.4f}")

        ckpt_path = os.path.join(args.out_dir, f"epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)

        if acc > best_acc:
            best_acc = acc
            best_path = os.path.join(args.out_dir, "best.pt")
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"  -> saved best: {best_path}")

if __name__ == "__main__":
    main()
