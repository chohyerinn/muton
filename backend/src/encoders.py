# Server/src/muton/encoders.py
from __future__ import annotations

import io
import json
import os
import re
import wave
import collections
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import cv2
import mediapipe as mp
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WavLMModel,
    pipeline,
)


# =========================
# Common
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Face Encoder (?먮낯 face.py 濡쒖쭅 洹몃?濡?
# =========================
@dataclass
class FaceConfig:
    resize_width: int = 256  # ?먮낯???덉뿀??:contentReference[oaicite:4]{index=4}


class FaceEncoder:
    def __init__(self, config: FaceConfig = None):
        self.config = config or FaceConfig()

        print("Loading Face Mesh (MediaPipe)...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        print("Loading Emotion Model (dima806/ViT)...")
        model_id = "dima806/facial_emotions_image_detection"
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).to(DEVICE).eval()

        self.mar_history = collections.deque(maxlen=5)
        self.MOVEMENT_THRESHOLD = 0.0003  # ?먮낯 洹몃?濡?:contentReference[oaicite:5]{index=5}

    def decode_jpeg(self, jpeg_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        # ???먮낯泥섎읆 90???뚯쟾 ?좎? :contentReference[oaicite:6]{index=6}
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def encode_jpeg_bytes(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        frame = self.decode_jpeg(jpeg_bytes)
        if frame is None:
            return {"status": "error", "reason": "decode_failed"}
        result = self.encode_frame(frame)
        if result is None:
            return {"status": "no_face"}
        result["status"] = "ok"
        return result

    def calculate_mar(self, landmarks):
        top = landmarks[13]
        bottom = landmarks[14]
        left = landmarks[61]
        right = landmarks[291]
        vertical = np.linalg.norm(np.array([top.x, top.y]) - np.array([bottom.x, bottom.y]))
        horizontal = np.linalg.norm(np.array([left.x, left.y]) - np.array([right.x, right.y]))
        if horizontal == 0:
            return 0.0
        return vertical / horizontal

    def align_face(self, frame, landmarks):
        img_h, img_w = frame.shape[:2]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        l_x, l_y = int(left_eye.x * img_w), int(left_eye.y * img_h)
        r_x, r_y = int(right_eye.x * img_w), int(right_eye.y * img_h)
        dy = r_y - l_y
        dx = r_x - l_x
        angle = np.degrees(np.arctan2(dy, dx))
        center = ((l_x + r_x) // 2, (l_y + r_y) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, M, (img_w, img_h), flags=cv2.INTER_CUBIC)

    @torch.no_grad()
    def encode_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        img_h, img_w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            self.mar_history.clear()
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # [A] speaking (mouth movement)
        mar = self.calculate_mar(landmarks)
        self.mar_history.append(mar)
        is_speaking = False
        variance = 0.0
        if len(self.mar_history) >= 3:
            variance = float(np.var(list(self.mar_history)))
            if variance > self.MOVEMENT_THRESHOLD:
                is_speaking = True

        # [B] align
        aligned_frame = self.align_face(frame_bgr, landmarks)

        # [C] crop (bbox + padding, ?먮낯 洹몃?濡? :contentReference[oaicite:7]{index=7}
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

        face_bgr = aligned_frame[y1:y2, x1:x2]
        if face_bgr is None or face_bgr.size == 0:
            return None

        if face_bgr.shape[0] < 10 or face_bgr.shape[1] < 10:
            return {
                "face_vec": [0.0] * 768,
                "face_emotion_logits": [0.0] * 7,
                "emotion": "Unknown",
                "emotion_probs": [],
                "is_speaking": is_speaking,
                "mar_variance": variance,
                "status": "ok",
            }

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # [D] ViT forward (embedding + logits) :contentReference[oaicite:8]{index=8}
        inputs = self.processor(images=face_rgb, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        # ??768 embedding (CLS)
        face_embedding = outputs.hidden_states[-1][:, 0, :].squeeze(0)  # (768,)
        # ??7 logits
        logits = outputs.logits.squeeze(0)  # (7,)

        probs = torch.softmax(logits, dim=0)
        scores = {
            self.model.config.id2label[i].lower(): float(probs[i])
            for i in range(len(probs))
        }

        # 媛먯젙 利앺룺 濡쒖쭅(?먮낯 ?좎?) :contentReference[oaicite:9]{index=9}
        final_emotion = "Neutral"
        if scores.get("surprise", 0) > 0.15:
            final_emotion = "Surprise"
        elif scores.get("angry", 0) > 0.230:
            final_emotion = "Angry"
        elif scores.get("disgust", 0) > 0.15:
            final_emotion = "Disgust"
        elif scores.get("sad", 0) > 0.20:
            final_emotion = "Sad"
        elif scores.get("fear", 0) > 0.20:
            final_emotion = "Fear"
        else:
            top_emotion = max(scores, key=scores.get)
            label_map = {
                "sad": "Sad",
                "disgust": "Disgust",
                "angry": "Angry",
                "neutral": "Neutral",
                "fear": "Fear",
                "surprise": "Surprise",
                "happy": "Happy",
            }
            final_emotion = label_map.get(top_emotion, top_emotion.capitalize())

        return {
            "face_vec": face_embedding.detach().cpu().tolist(),          # (768,)
            "face_emotion_logits": logits.detach().cpu().tolist(),       # (7,)
            "emotion": final_emotion,
            "emotion_probs": probs.detach().cpu().tolist(),
            "is_speaking": is_speaking,
            "mar_variance": variance,
            "status": "ok",
        }


# =========================
# Audio Encoder (?먮낯 audio.py 濡쒖쭅 洹몃?濡?+ ?ㅻ쭔 env濡?
# =========================
class AudioEncoder:
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.sample_rate = 16000
        self.runtime_stt_backend = os.environ.get("MUTON_QWEN_STT_BACKEND", "whisper").strip().lower()

        self.audio_buffer = bytearray()
        self.stt_model_name = os.environ.get(
            "MUTON_STT_MODEL_NAME",
            "ghost613/whisper-large-v3-turbo-korean",
        ).strip()
        self.stt_device = os.environ.get("MUTON_STT_DEVICE", self.device).strip()
        self.stt_language = os.environ.get("MUTON_STT_LANGUAGE", "ko").strip()
        self.stt_prompt = os.environ.get(
            "MUTON_STT_PROMPT",
            "Transcribe Korean speech faithfully as subtitles. Output only the spoken utterance.",
        ).strip()
        self.openai_stt_model = os.environ.get("MUTON_OPENAI_STT_MODEL", "whisper-1").strip()
        self.openai_stt_prompt = os.environ.get("MUTON_OPENAI_STT_PROMPT", "대화 내용입니다. 핵심 내용만 적으세요.").strip()
        self.stt_max_new_tokens = int(os.environ.get("MUTON_STT_MAX_NEW_TOKENS", "64").strip())
        self.stt_vad_threshold = float(os.environ.get("MUTON_STT_VAD_THRESHOLD", "0.85").strip())
        self.min_energy_threshold = int(os.environ.get("MUTON_STT_MIN_CHUNK_ENERGY", "700").strip())
        self.min_buffer_bytes = int(os.environ.get("MUTON_STT_MIN_BUFFER_BYTES", "40000").strip())
        self.max_buffer_bytes = int(os.environ.get("MUTON_STT_MAX_BUFFER_BYTES", "320000").strip())
        self.min_duration_bytes = int(os.environ.get("MUTON_STT_MIN_DURATION_BYTES", "40000").strip())
        self.min_full_energy = int(os.environ.get("MUTON_STT_MIN_UTTERANCE_ENERGY", "450").strip())
        self.max_silence_chunks = int(os.environ.get("MUTON_STT_MAX_SILENCE_CHUNKS", "7").strip())
        self.max_repeat_tokens = int(os.environ.get("MUTON_STT_MAX_REPEAT_TOKENS", "3").strip())
        self.min_transcript_confidence = float(
            os.environ.get("MUTON_STT_MIN_TRANSCRIPT_CONFIDENCE", "0.45").strip()
        )
        stt_dtype_name = os.environ.get(
            "MUTON_STT_TORCH_DTYPE",
            "float16" if self.stt_device.startswith("cuda") else "float32",
        ).strip()
        self.stt_torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(stt_dtype_name, torch.float16 if self.stt_device.startswith("cuda") else torch.float32)

        self.stt_processor = None
        self.stt_model = None
        self.stt_pipe = None
        if self.runtime_stt_backend != "openai":
            print(f"Loading Korean Whisper STT ({self.stt_model_name})...")
            self.stt_processor = AutoProcessor.from_pretrained(self.stt_model_name)
            self.stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.stt_model_name,
                torch_dtype=self.stt_torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            ).to(self.stt_device).eval()
            # Older Whisper checkpoints often carry max_length=20 in generation config,
            # which triggers noisy warnings when we drive decoding with max_new_tokens.
            try:
                self.stt_model.generation_config.max_length = None
                self.stt_model.generation_config.max_new_tokens = None
            except Exception:
                pass
            try:
                self.stt_processor.tokenizer.set_prefix_tokens(
                    language=self.stt_language,
                    task="transcribe",
                )
            except Exception:
                pass

            pipeline_device = -1
            if self.stt_device.startswith("cuda"):
                pipeline_device = int(self.stt_device.split(":", 1)[1]) if ":" in self.stt_device else 0

            self.stt_pipe = pipeline(
                "automatic-speech-recognition",
                model=self.stt_model,
                tokenizer=self.stt_processor.tokenizer,
                feature_extractor=self.stt_processor.feature_extractor,
                torch_dtype=self.stt_torch_dtype,
                device=pipeline_device,
            )
            try:
                self.stt_pipe.model.generation_config.max_length = None
                self.stt_pipe.model.generation_config.max_new_tokens = None
            except Exception:
                pass
        else:
            print("Skipping local Korean Whisper STT load because MUTON_QWEN_STT_BACKEND=openai")

        print("Loading Silero VAD...")
        self.vad_model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.vad_model.to(self.device)

        self.speech_threshold = self.stt_vad_threshold
        self.silence_chunks = 0
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.openai_client = None
        if api_key and OpenAI is not None:
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception:
                self.openai_client = None

        self.noise_words = [
            "MBC 뉴스",
            "MBC뉴스",
            "시청해주셔서 감사합니다",
            "구독과 좋아요",
            "알림 설정",
            "YTN",
            "KBS",
            "SBS",
            "유료광고",
            "기자",
            "보도국",
            "투데이 별별영상",
            "자막뉴스",
            "포함하고 있습니다",
            "주식시황",
            "오늘의 주식",
            "뉴스 스토리",
            "이덕영",
        ]
        self.filler_words = ["음...", "음", "어...", "어", "그...", "그", "아...", "아", "저...", "저", "에...", "에", "으음"]

        print("Loading WavLM-base-plus...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(self.device).eval()

    def stt_with_api(self, raw_bytes: bytes) -> Optional[str]:
        transcript, _, _ = self.consume_buffered_speech_openai(raw_bytes)
        return transcript

    def _prepare_buffered_utterance(self, raw_bytes: bytes) -> tuple[Optional[np.ndarray], float]:
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        if len(audio_int16) > 0:
            chunk_energy = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
        else:
            chunk_energy = 0.0

        if len(self.audio_buffer) == 0 and chunk_energy < self.min_energy_threshold:
            return None, 0.0

        self.audio_buffer.extend(raw_bytes)

        is_speech = False
        if chunk_energy > self.min_energy_threshold:
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            window_size = 512
            for i in range(0, len(audio_float32), window_size):
                chunk = audio_float32[i : i + window_size]
                if len(chunk) < window_size:
                    break
                tensor_chunk = torch.from_numpy(chunk).to(self.device).unsqueeze(0)
                speech_prob = self.vad_model(tensor_chunk, 16000).item()
                if speech_prob > self.speech_threshold:
                    is_speech = True
                    break

        if is_speech:
            self.silence_chunks = 0
        else:
            self.silence_chunks += 1

        should_send = False
        if len(self.audio_buffer) > self.min_buffer_bytes and self.silence_chunks > self.max_silence_chunks:
            should_send = True
        elif len(self.audio_buffer) > self.max_buffer_bytes:
            should_send = True
            print("Detected: force send (buffer full)")

        if not should_send:
            return None, 0.0

        if len(self.audio_buffer) < self.min_duration_bytes:
            print(f"Discarded because the chunk is too short (size={len(self.audio_buffer)})")
            self.audio_buffer = bytearray()
            self.silence_chunks = 0
            return None, 0.0

        full_buffer_int16 = np.frombuffer(self.audio_buffer, dtype=np.int16)
        full_energy = np.sqrt(np.mean(full_buffer_int16.astype(np.float32) ** 2))
        if full_energy < self.min_full_energy:
            print(f"Discarded because full energy is too low (energy={int(full_energy)})")
            self.audio_buffer = bytearray()
            self.silence_chunks = 0
            return None, 0.0

        utterance_bytes = bytes(self.audio_buffer)
        waveform = np.frombuffer(utterance_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer = bytearray()
        self.silence_chunks = 0
        return waveform, float(full_energy)

    def _transcribe_with_openai(self, waveform: np.ndarray) -> tuple[Optional[str], float]:
        if self.openai_client is None or waveform.size == 0:
            return None, 0.0

        wav_io = io.BytesIO()
        wav_io.name = "speech.wav"
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            pcm = np.clip(waveform * 32768.0, -32768, 32767).astype(np.int16)
            wav_file.writeframes(pcm.tobytes())
        wav_io.seek(0)

        try:
            transcript = self.openai_client.audio.transcriptions.create(
                model=self.openai_stt_model,
                file=wav_io,
                language=self.stt_language,
                response_format="verbose_json",
                prompt=self.openai_stt_prompt,
                temperature=0.0,
            )
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return None, 0.0

        raw_text = (
            getattr(transcript, "text", None)
            or (transcript.get("text") if isinstance(transcript, dict) else "")
            or ""
        ).strip()

        segments = getattr(transcript, "segments", None)
        if segments is None and isinstance(transcript, dict):
            segments = transcript.get("segments", None)

        avg_logprob = None
        no_speech_prob = None
        if segments:
            seg0 = segments[0]
            if isinstance(seg0, dict):
                avg_logprob = seg0.get("avg_logprob")
                no_speech_prob = seg0.get("no_speech_prob")
            else:
                avg_logprob = getattr(seg0, "avg_logprob", None)
                no_speech_prob = getattr(seg0, "no_speech_prob", None)

        if avg_logprob is not None and avg_logprob < -1.0:
            print(f"Discarded because avg_logprob is too low ({avg_logprob:.2f}): {raw_text}")
            return None, 0.0
        if no_speech_prob is not None and no_speech_prob > 0.8:
            print(f"Discarded because no_speech_prob is too high ({no_speech_prob:.2f}): {raw_text}")
            return None, 0.0

        confidence = 0.5
        if avg_logprob is not None:
            confidence = 0.5 * float(np.clip((avg_logprob + 1.5) / 1.5, 0.0, 1.0)) + 0.5 * confidence
        if no_speech_prob is not None:
            confidence = 0.5 * confidence + 0.5 * (1.0 - float(no_speech_prob))

        return raw_text or None, float(np.clip(confidence, 0.0, 1.0))

    def _estimate_transcript_confidence(
        self,
        text: str,
        waveform: np.ndarray,
        full_energy: float,
    ) -> float:
        if not text or waveform.size == 0:
            return 0.0

        tokens = text.split()
        duration_sec = waveform.size / self.sample_rate
        unique_ratio = len(set(tokens)) / max(len(tokens), 1) if tokens else 0.0
        hangul_chars = sum(1 for ch in text if "\uac00" <= ch <= "\ud7a3")
        alnum_chars = sum(1 for ch in text if ch.isalnum())
        hangul_ratio = min(1.0, hangul_chars / max(alnum_chars, 1))

        energy_score = float(np.clip((full_energy - self.min_full_energy) / max(1, 1800 - self.min_full_energy), 0.0, 1.0))
        duration_score = float(np.clip(duration_sec / 2.0, 0.0, 1.0))
        token_score = float(np.clip(len(tokens) / 5.0, 0.0, 1.0))

        confidence = (
            0.30 * energy_score
            + 0.20 * duration_score
            + 0.20 * token_score
            + 0.20 * unique_ratio
            + 0.10 * hangul_ratio
        )

        if len(tokens) <= 1 and len(text) <= 4:
            confidence *= 0.6

        return float(np.clip(confidence, 0.0, 1.0))

    def _filter_transcript(self, raw_text: str) -> Optional[str]:
        if not raw_text:
            return None

        filtered_text = raw_text
        for noise in self.noise_words:
            filtered_text = filtered_text.replace(noise, "")
        for filler in self.filler_words:
            filtered_text = filtered_text.replace(f"{filler} ", "").replace(f" {filler}", "")
            if filtered_text == filler:
                filtered_text = ""

        filtered_text = filtered_text.strip()
        filtered_text = re.sub(r"^[,\.]+", "", filtered_text).strip()
        filtered_text = re.sub(r"\s+", " ", filtered_text)

        repeated_char_match = re.fullmatch(r"(.{1,2})\1{2,}", filtered_text)
        if repeated_char_match:
            return None
        repeated_korean_span = re.search(r"([가-힣]{1,4})\1{2,}", filtered_text)
        if repeated_korean_span:
            return None

        raw_tokens = filtered_text.split()
        if raw_tokens:
            longest_repeat = 1
            current_repeat = 1
            for prev_token, token in zip(raw_tokens, raw_tokens[1:]):
                if token == prev_token:
                    current_repeat += 1
                    longest_repeat = max(longest_repeat, current_repeat)
                else:
                    current_repeat = 1
            if longest_repeat >= self.max_repeat_tokens:
                return None

        tokens = raw_tokens
        if tokens:
            collapsed_tokens: list[str] = []
            prev_token = None
            for token in tokens:
                if token == prev_token:
                    continue
                collapsed_tokens.append(token)
                prev_token = token
            tokens = collapsed_tokens
            filtered_text = " ".join(tokens)

        if tokens:
            max_count = max(tokens.count(token) for token in set(tokens))
            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            if len(tokens) >= 4 and max_count / len(tokens) >= 0.6:
                return None
            if len(tokens) >= 6 and unique_ratio < 0.4:
                return None

        if any(x in raw_text for x in ["유료광고", "구독", "기자", "뉴스"]):
            return None
        if len(filtered_text) < 2 and not any(c.isalnum() for c in filtered_text):
            return None

        return filtered_text or None

    def _transcribe_waveform(self, waveform: np.ndarray) -> Optional[str]:
        if waveform.size == 0 or self.stt_pipe is None:
            return None

        try:
            result = self.stt_pipe(
                {"array": waveform.astype(np.float32, copy=False), "sampling_rate": self.sample_rate},
                max_length=self.stt_max_new_tokens,
                return_timestamps=False,
            )
        except Exception as e:
            print(f"Local Whisper STT Error: {e}")
            return None

        if isinstance(result, dict):
            return (result.get("text") or "").strip()
        return str(result).strip()

    def consume_buffered_speech(self, raw_bytes: bytes) -> tuple[Optional[str], Optional[np.ndarray], float]:
        waveform, full_energy = self._prepare_buffered_utterance(raw_bytes)
        if waveform is None:
            return None, None, 0.0

        raw_text = self._transcribe_waveform(waveform)
        filtered_text = self._filter_transcript(raw_text or "")
        confidence = self._estimate_transcript_confidence(filtered_text or "", waveform, float(full_energy))
        if filtered_text and confidence < self.min_transcript_confidence:
            print(f"Discarded because transcript confidence is too low (conf={confidence:.2f}, text={filtered_text})")
            return None, waveform, confidence
        if filtered_text:
            print(f"Local Whisper transcript: {filtered_text} (conf={confidence:.2f})")
        return filtered_text, waveform, confidence

    def consume_buffered_speech_openai(self, raw_bytes: bytes) -> tuple[Optional[str], Optional[np.ndarray], float]:
        waveform, full_energy = self._prepare_buffered_utterance(raw_bytes)
        if waveform is None:
            return None, None, 0.0

        raw_text, api_confidence = self._transcribe_with_openai(waveform)
        filtered_text = self._filter_transcript(raw_text or "")
        # For the OpenAI Whisper path, trust the API confidence signal and
        # keep the old behavior closer to the initial implementation.
        confidence = api_confidence if filtered_text else 0.0
        if filtered_text:
            print(f"OpenAI Whisper transcript: {filtered_text} (conf={confidence:.2f})")
        return filtered_text, waveform, confidence

    # WavLM feature extractor (?먮낯 洹몃?濡? :contentReference[oaicite:10]{index=10}
    def extract_features_from_pcm(self, pcm_np: np.ndarray) -> Dict[str, Any]:
        with torch.no_grad():
            wav = torch.tensor(pcm_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.wavlm(wav, output_hidden_states=True)
            hidden = out.last_hidden_state
            T = hidden.size(1)

            content = hidden.mean(dim=1).squeeze(0)

            sec = len(pcm_np) / self.sample_rate
            frames_per_sec = T / sec if sec > 0 else T
            n = int(frames_per_sec)
            if n > T:
                n = T
            speaker = hidden[:, -n:, :].mean(dim=1).squeeze(0)

            wav_cpu = wav.squeeze(0).cpu()
            frame_len = int(0.025 * 16000)
            hop = int(0.010 * 16000)
            energies = [
                wav_cpu[i: i + frame_len].abs().mean().item()
                for i in range(0, wav_cpu.numel() - frame_len, hop)
            ]
            if len(energies) == 0:
                energies = np.ones(1)

            x_old = np.linspace(0, 1, len(energies))
            x_new = np.linspace(0, 1, T)
            e_interp = np.interp(x_new, x_old, energies)
            w = e_interp / (e_interp.sum() + 1e-9)
            w_t = torch.tensor(w, device=self.device).unsqueeze(1)
            prosody = (hidden.squeeze(0) * w_t).sum(dim=0)

            return {
                "prosody": prosody.cpu().numpy().tolist(),
                "content": content.cpu().numpy().tolist(),
                "speaker": speaker.cpu().numpy().tolist(),
            }


# =========================
# Text Encoder (embedding.py?먯꽌 ?곕뜕 mean-pool 洹몃?濡??섑븨)
# =========================
class TextEncoder:
    def __init__(self, model_name: str = "klue/roberta-small", device: str = DEVICE, max_length: int = 64):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).to(self.device)

        out = self.model(**inputs, return_dict=True)
        last_hidden = out.last_hidden_state
        mask = inputs["attention_mask"]

        mask_f = mask.unsqueeze(-1).float()
        summed = (last_hidden * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1e-6)
        vec = (summed / denom).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return vec  # (768,)
    
    @staticmethod
    def load_translate_cache(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def save_translate_cache(path: str, cache: dict):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def get_korean_script(sample_id: str, utterance_en: str, cache: dict) -> str:
        # 1) 罹먯떆???덉쑝硫?洹멸굅 ?
        if sample_id in cache:
            return cache[sample_id]

        # 2) ?놁쑝硫??쇰떒 ?곸뼱 洹몃?濡?=placeholder)
        # TODO: ?ш린??OpenAI/濡쒖뺄踰덉뿭 ?몄텧濡?諛붽씀硫???
        ko = utterance_en

        cache[sample_id] = ko
        return ko

