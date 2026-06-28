from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput


def normalize_fusion_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map older encoder-style checkpoint names onto the current backbone names."""
    if "head_emo.weight" not in state_dict:
        return state_dict
    if not any(key.startswith("encoder.layers.") for key in state_dict):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            layer_idx = parts[2]
            suffix = ".".join(parts[3:])

            if suffix.startswith("self_attn."):
                new_key = f"layers.{layer_idx}.{suffix[len('self_attn.'):]}"
            elif suffix.startswith("linear1."):
                new_key = f"ffns.{layer_idx}.0.{suffix[len('linear1.'):]}"
            elif suffix.startswith("linear2."):
                new_key = f"ffns.{layer_idx}.3.{suffix[len('linear2.'):]}"
            elif suffix.startswith("norm1."):
                new_key = f"norms1.{layer_idx}.{suffix[len('norm1.'):]}"
            elif suffix.startswith("norm2."):
                new_key = f"norms2.{layer_idx}.{suffix[len('norm2.'):]}"

        remapped[new_key] = value

    return remapped


def load_compatible_fusion_weights(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str]]:
    """
    Load only the subset of checkpoint weights that match by both name and shape.
    This lets us reuse the fusion backbone while changing the class count or decoder.
    """
    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in normalize_fusion_state_dict(state_dict).items():
        if key not in model_state:
            skipped.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing), skipped + list(unexpected)


class FusionEncoderDecoder(nn.Module):
    """
    Fusion encoder + seq2seq decoder.

    The encoder keeps the current MUTON multimodal tokenization:
    [CLS, face_vec, face_emo, audio_content, audio_speaker, audio_prosody, text]

    The decoder cross-attends directly to those fused hidden states instead of
    receiving a prompt that only contains predicted class/arousal/valence values.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        nlayers: int = 4,
        dropout: float = 0.1,
        num_emotions: int = 6,
        decoder_model_name: str = "google/mt5-small",
    ) -> None:
        super().__init__()

        self.proj_face = nn.Linear(768, d_model)
        self.proj_a_cont = nn.Linear(768, d_model)
        self.proj_a_spk = nn.Linear(768, d_model)
        self.proj_a_pros = nn.Linear(768, d_model)
        self.proj_text = nn.Linear(768, d_model)
        self.proj_faceemo = nn.Linear(7, d_model)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                for _ in range(nlayers)
            ]
        )
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nlayers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nlayers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                )
                for _ in range(nlayers)
            ]
        )

        self.head_emo = nn.Linear(d_model, num_emotions)
        self.head_arousal = nn.Linear(d_model, 1)
        self.head_valence = nn.Linear(d_model, 1)

        self.decoder = AutoModelForSeq2SeqLM.from_pretrained(decoder_model_name)
        decoder_cfg = AutoConfig.from_pretrained(decoder_model_name)
        decoder_hidden = getattr(decoder_cfg, "d_model", None) or getattr(decoder_cfg, "hidden_size", None)
        if decoder_hidden is None:
            raise ValueError(f"Could not infer decoder hidden size from {decoder_model_name}")

        self.memory_proj = nn.Linear(d_model, decoder_hidden) if d_model != decoder_hidden else nn.Identity()
        self.memory_norm = nn.LayerNorm(decoder_hidden)
        self.decoder_model_name = decoder_model_name

    def encode_modalities(
        self,
        batch: Dict[str, torch.Tensor],
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = batch["face_vec"].shape[0]
        t0 = self.proj_face(batch["face_vec"])
        t1 = self.proj_faceemo(batch["face_emo"])
        t2 = self.proj_a_cont(batch["a_cont"])
        t3 = self.proj_a_spk(batch["a_spk"])
        t4 = self.proj_a_pros(batch["a_pros"])
        t5 = self.proj_text(batch["text"])

        tokens = torch.stack([t0, t1, t2, t3, t4, t5], dim=1)
        cls = self.cls.expand(batch_size, -1, -1)
        x = torch.cat([cls, tokens], dim=1)

        attn_last = None
        for attn, norm1, norm2, ffn in zip(self.layers, self.norms1, self.norms2, self.ffns):
            attn_out, attn_weights = attn(
                x,
                x,
                x,
                need_weights=return_attn,
                average_attn_weights=False,
            )
            x = norm1(x + attn_out)
            x = norm2(x + ffn(x))
            if return_attn:
                attn_last = attn_weights

        cls_vec = x[:, 0]
        cls_attn = None
        if return_attn and attn_last is not None:
            cls_attn = attn_last.mean(1)[:, 0, 1:]

        return x, cls_vec, cls_attn

    def _prepare_decoder_memory(self, memory: torch.Tensor) -> Tuple[BaseModelOutput, torch.Tensor]:
        projected = self.memory_norm(self.memory_proj(memory))
        attention_mask = torch.ones(
            projected.shape[:2],
            dtype=torch.long,
            device=projected.device,
        )
        return BaseModelOutput(last_hidden_state=projected), attention_mask

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> Dict[str, torch.Tensor | None]:
        memory, cls_vec, cls_attn = self.encode_modalities(batch, return_attn=return_attn)

        emo_logits = self.head_emo(cls_vec)
        arousal = self.head_arousal(cls_vec).squeeze(-1)
        valence = self.head_valence(cls_vec).squeeze(-1)

        outputs: Dict[str, torch.Tensor | None] = {
            "memory": memory,
            "emo_logits": emo_logits,
            "arousal": arousal,
            "valence": valence,
            "cls_attn": cls_attn,
            "gen_loss": None,
            "decoder_logits": None,
        }

        if labels is not None:
            encoder_outputs, attention_mask = self._prepare_decoder_memory(memory)
            decoder_out = self.decoder(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            outputs["gen_loss"] = decoder_out.loss
            outputs["decoder_logits"] = decoder_out.logits

        return outputs

    @torch.no_grad()
    def generate(
        self,
        batch: Dict[str, torch.Tensor],
        **generate_kwargs,
    ) -> torch.Tensor:
        memory, _, _ = self.encode_modalities(batch, return_attn=False)
        encoder_outputs, attention_mask = self._prepare_decoder_memory(memory)
        return self.decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def freeze_encoder(self) -> None:
        for name, param in self.named_parameters():
            if not name.startswith("decoder."):
                param.requires_grad = False

    def freeze_decoder(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder_last_nlayers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        start = max(0, len(self.layers) - n_layers)
        for layer_idx in range(start, len(self.layers)):
            for param in self.layers[layer_idx].parameters():
                param.requires_grad = True
            for param in self.norms1[layer_idx].parameters():
                param.requires_grad = True
            for param in self.norms2[layer_idx].parameters():
                param.requires_grad = True
            for param in self.ffns[layer_idx].parameters():
                param.requires_grad = True
