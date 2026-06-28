# -*- coding: utf-8 -*-
import os
import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_fusion_meld import FusionTransformer  # backbone 구조 그대로 씀

# ===== 너 한국어 라벨(소문자) =====
KO_EMO2ID = {
    "angry": 0,
    "dislike": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5,
}
KO_ID2EMO = {v: k for k, v in KO_EMO2ID.items()}

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

    emo = torch.tensor([KO_EMO2ID.get(str(b["emotion"]).strip().lower(), KO_EMO2ID["neutral"]) for b in batch],
                       dtype=torch.long)
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
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def freeze_backbone(model: FusionTransformer, unfreeze_last_nlayers: int = 0):
    """
    head만 학습이 기본.
    unfreeze_last_nlayers>0이면 encoder의 마지막 n layer만 풀어서 미세조정 가능(데이터 적으면 보통 0 추천)
    """
    # 전부 freeze
    for p in model.parameters():
        p.requires_grad = False

    # heads는 학습
    for name, p in model.named_parameters():
        if "head_" in name:
            p.requires_grad = True

    # encoder 일부 unfreeze 옵션
    if unfreeze_last_nlayers > 0:
        # TransformerEncoder는 .layers에 접근 가능
        layers = getattr(model, "layers", None)
        if layers is None and hasattr(model, "encoder"):
            layers = model.encoder.layers
        if layers is None:
            raise AttributeError("FusionTransformer does not expose transformer layers.")
        n = len(layers)
        for li in range(max(0, n - unfreeze_last_nlayers), n):
            for p in layers[li].parameters():
                p.requires_grad = True

        # projection도 같이 조금 풀어주고 싶으면 여기서 추가 가능(지금은 안 푼다)

@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    ys, ps = [], []
    mse_a = 0.0
    mse_v = 0.0
    n = 0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        emo_logits, a_pred, v_pred = model(batch)

        y = batch["emo"]
        p = emo_logits.argmax(dim=-1)

        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(p.detach().cpu().numpy().tolist())

        mse_a += F.mse_loss(a_pred, batch["arousal"], reduction="sum").item()
        mse_v += F.mse_loss(v_pred, batch["valence"], reduction="sum").item()
        n += y.numel()

    ys = np.array(ys, dtype=np.int64)
    ps = np.array(ps, dtype=np.int64)
    acc = float((ys == ps).mean()) if len(ys) else 0.0
    return acc, mse_a / max(1, n), mse_v / max(1, n), ys, ps

def make_folds(n, k, seed=42):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    return folds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_ckpt", type=str, required=True)      # MELD best.pt
    ap.add_argument("--ko_pt", type=str, required=True)         # fusion_dataset.pt (30개)
    ap.add_argument("--out_dir", type=str, default="out/fusion_ko_kfold")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unfreeze_last_nlayers", type=int, default=0)
    ap.add_argument("--w_emo", type=float, default=1.0)
    ap.add_argument("--w_arousal", type=float, default=0.5)
    ap.add_argument("--w_valence", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(args.ko_pt)
    print("KO N =", len(data))
    print("KO label dist:", Counter([str(x["emotion"]).strip().lower() for x in data]))

    # load pretrained config
    pkg = torch.load(args.pre_ckpt, map_location="cpu")
    cfg = pkg.get("args", {})
    backbone_cfg = {
        "d_model": cfg.get("d_model", 256),
        "nhead": cfg.get("nhead", 8),
        "nlayers": cfg.get("nlayers", 4),
    }

    folds = make_folds(len(data), args.k, seed=args.seed)

    fold_scores = []
    for fi in range(args.k):
        test_idx = set(folds[fi])
        train_data = [data[i] for i in range(len(data)) if i not in test_idx]
        test_data  = [data[i] for i in range(len(data)) if i in test_idx]

        train_ds = ListDataset(train_data)
        test_ds  = ListDataset(test_data)

        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0, collate_fn=ko_collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, num_workers=0, collate_fn=ko_collate_fn)

        # model build: 먼저 MELD 구조(7)로 만들고 로드
        model = FusionTransformer(
            d_model=backbone_cfg["d_model"],
            nhead=backbone_cfg["nhead"],
            nlayers=backbone_cfg["nlayers"],
            num_emotions=7,
        ).to(device)
        model.load_state_dict(pkg["model"], strict=True)

        # head 교체 (6-class)
        d_in = model.head_emo.in_features
        model.head_emo = nn.Linear(d_in, len(KO_EMO2ID)).to(device)

        freeze_backbone(model, unfreeze_last_nlayers=args.unfreeze_last_nlayers)

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

        best_acc = -1.0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            steps = 0
            for batch in train_loader:
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)

                emo_logits, a_pred, v_pred = model(batch)
                loss_emo = F.cross_entropy(emo_logits, batch["emo"])
                loss_a = F.mse_loss(a_pred, batch["arousal"])
                loss_v = F.mse_loss(v_pred, batch["valence"])
                loss = args.w_emo * loss_emo + args.w_arousal * loss_a + args.w_valence * loss_v

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

                total += loss.item()
                steps += 1

            acc, mse_a, mse_v, _, _ = eval_one(model, test_loader, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 20 == 0 or epoch == 1:
                print(f"[fold {fi+1}/{args.k} epoch {epoch}] train_loss={total/max(1,steps):.4f} test_acc={acc:.4f} mse_a={mse_a:.4f} mse_v={mse_v:.4f}")

        # save fold best
        ckpt = {
            "model": best_state,
            "pre_ckpt": args.pre_ckpt,
            "ko_emo2id": KO_EMO2ID,
            "args": vars(args),
            "backbone_cfg": backbone_cfg,
            "fold": fi,
            "best_acc": best_acc,
        }
        fold_path = os.path.join(args.out_dir, f"fold{fi}_best.pt")
        torch.save(ckpt, fold_path)
        print(f"[fold {fi+1}] BEST acc={best_acc:.4f} saved={fold_path}")

        fold_scores.append(best_acc)

    print("\n===== K-FOLD RESULT =====")
    print("accs:", [round(x, 4) for x in fold_scores])
    print("mean_acc:", float(np.mean(fold_scores)))
    print("std_acc:", float(np.std(fold_scores)))

if __name__ == "__main__":
    main()
