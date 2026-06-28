# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

from train_fusion_meld import FusionTransformer, PTDictDataset, collate_fn, EMO2ID
ID2EMO = {v: k for k, v in EMO2ID.items()}

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

@torch.no_grad()
def run_eval_collect(model, loader, device):
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
    mse_a = mse_a / max(1, n)
    mse_v = mse_v / max(1, n)
    return acc, mse_a, mse_v, ys, ps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--test_pt", type=str, required=True)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pkg = torch.load(args.ckpt, map_location="cpu")
    cfg = pkg.get("args", {})

    model = FusionTransformer(
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        nlayers=cfg.get("nlayers", 4),
        num_emotions=7,
    ).to(device)
    model.load_state_dict(pkg["model"], strict=True)

    ds = PTDictDataset(args.test_pt)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn
    )

    acc, mse_a, mse_v, ys, ps = run_eval_collect(model, loader, device)

    print(f"TEST: acc={acc:.4f} mse_a={mse_a:.4f} mse_v={mse_v:.4f}")

    n_cls = 7
    cm = confusion_matrix(ys, ps, n_cls)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    stats = per_class_stats(cm)
    print("\nPer-class (Precision, Recall, F1):")
    for i in range(n_cls):
        name = ID2EMO[i]
        p, r, f1 = stats[i]
        support = cm[i, :].sum()
        print(f"{i:2d} {name:10s} P={p:.3f} R={r:.3f} F1={f1:.3f} support={support}")

    print("\nTRUE dist:", Counter(ys.tolist()))
    print("PRED dist:", Counter(ps.tolist()))

if __name__ == "__main__":
    main()
