import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from muton.config import env_path


DATA_PATH = env_path("MUTON_FUSION_DATASET", "data/fusion_dataset.pt")
MODEL_SAVE_PATH = env_path("MUTON_SELFATTN_MODEL", "artifacts/my_transformer_fusion.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 50
LR = 5e-5

EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "surprise": 4,
    "fear": 5,
    "disgust": 6,
    "dislike": 0,
}


class FusionDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        emo = EMOTION_MAP.get(str(item.get("emotion", "neutral")).lower(), 0)
        aro = float(item.get("arousal", 5.0))
        val = float(item.get("valence", 5.0))

        return (
            item["face_vec"],
            item["face_emo_logits"],
            item["audio_content"],
            item["audio_speaker"],
            item["audio_prosody"],
            item["text"],
            torch.tensor(emo, dtype=torch.long),
            torch.tensor(aro, dtype=torch.float32),
            torch.tensor(val, dtype=torch.float32),
        )


class FusionBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


class MultimodalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_classes=7):
        super().__init__()
        self.face_proj = nn.Linear(768, d_model)
        self.face_emo_proj = nn.Linear(7, d_model)
        self.audio_c_proj = nn.Linear(768, d_model)
        self.audio_s_proj = nn.Linear(768, d_model)
        self.audio_p_proj = nn.Linear(768, d_model)
        self.text_proj = nn.Linear(768, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.block = FusionBlock(d_model=d_model, nhead=nhead)
        self.emotion_head = nn.Linear(d_model, num_classes)
        self.arousal_head = nn.Linear(d_model, 1)
        self.valence_head = nn.Linear(d_model, 1)

    def forward(self, fv, fe, ac, asp, apr, text, return_attn=False):
        batch_size = fv.size(0)
        tokens = torch.stack(
            [
                self.face_proj(fv),
                self.face_emo_proj(fe),
                self.audio_c_proj(ac),
                self.audio_s_proj(asp),
                self.audio_p_proj(apr),
                self.text_proj(text),
            ],
            dim=1,
        )
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out, attn_weights = self.block(seq)
        cls_out = out[:, 0]

        pred_emo = self.emotion_head(cls_out)
        pred_aro = self.arousal_head(cls_out)
        pred_val = self.valence_head(cls_out)

        if return_attn:
            return pred_emo, pred_aro, pred_val, attn_weights
        return pred_emo, pred_aro, pred_val


def train():
    dataset = FusionDataset(str(DATA_PATH))
    if len(dataset) > 50:
        train_size = int(0.9 * len(dataset))
        train_data, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    else:
        train_data = dataset

    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    model = MultimodalTransformer().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    print("Training multimodal fusion transformer...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for fv, fe, ac, asp, apr, text, labels, aro, val in loader:
            fv, fe, ac, asp, apr, text = (
                fv.to(DEVICE),
                fe.to(DEVICE),
                ac.to(DEVICE),
                asp.to(DEVICE),
                apr.to(DEVICE),
                text.to(DEVICE),
            )
            labels = labels.to(DEVICE)
            aro = aro.to(DEVICE).unsqueeze(1)
            val = val.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            pred_emo, pred_aro, pred_val = model(fv, fe, ac, asp, apr, text)
            loss = (
                criterion_cls(pred_emo, labels)
                + 0.5 * criterion_reg(pred_aro, aro)
                + 0.5 * criterion_reg(pred_val, val)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(loader):.4f}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
    print(f"Model saved -> {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
