# =====================================================
# server.py (주석 풀버전)
# - Android(카메라/마이크) → FastAPI 서버로 chunk 전송
# - 서버는 (1) 얼굴 인코딩/감정 (2) STT+오디오 특징 (3) 멀티모달 fusion + GPT 요약
# =====================================================

# ---------------------------
# 기본 라이브러리
# ---------------------------
import json                          # JSON 문자열 ↔ 파이썬 객체 변환 (Form으로 받은 벡터 파싱 등에 사용)
import random                        # few-shot 예시 랜덤 샘플링에 사용
import numpy as np                   # attention 기반 temperature 계산, pcm → float 변환 등에 사용
import torch                         # 모델 로딩/추론 (PyTorch)
import torch.nn as nn                # FusionBlock / Linear / LayerNorm 등 모델 구성
import uvicorn                       # FastAPI 서버 실행기
import time                          # 얼굴 캐시 타임스탬프 관리(최근 프레임 유지)

# ---------------------------
# FastAPI 관련
# ---------------------------
from fastapi import FastAPI, UploadFile, File, Form  # API 서버/멀티파트 업로드/폼데이터 입력
from fastapi.middleware.cors import CORSMiddleware   # Android 앱에서 CORS 막힘 방지
from fastapi.responses import JSONResponse           # (필요시) JSON 응답 커스텀

# ---------------------------
# Transformers(텍스트 임베딩)
# ---------------------------
from transformers import AutoTokenizer, AutoModel    # KLUE RoBERTa 로딩

# ---------------------------
# 프로젝트 내부 모듈
# ---------------------------
from muton.config import env_path
from muton.encoders import FaceEncoder, AudioEncoder, TextEncoder

# =====================================================
# Global Face Cache
# - video endpoint가 계속 들어오므로
#   가장 최근 얼굴 벡터(latest_face_vec)를 전역에 저장해두고
#   audio 분석이 들어오면 그때 "가장 최근 얼굴"을 같이 사용함
# =====================================================
latest_face_vec = None               # 최신 얼굴 결과(dict): face_vec(768), face_emotion_logits(7), emotion label 등
face_vec_timestamp = 0               # latest_face_vec가 갱신된 시각(초 단위 epoch time)

# =====================================================
# DEVICE 설정
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 있으면 CUDA 사용, 없으면 CPU

# =====================================================
# FastAPI 앱 생성 + CORS 허용
# =====================================================
app = FastAPI()  # FastAPI 인스턴스 생성

app.add_middleware(
    CORSMiddleware,                 # CORS 미들웨어 활성화
    allow_origins=["*"],            # 모든 도메인 허용 (개발용; 배포 시 특정 도메인으로 제한 권장)
    allow_credentials=True,         # 쿠키/인증 포함 요청 허용
    allow_methods=["*"],            # GET/POST 등 모든 메서드 허용
    allow_headers=["*"],            # 모든 헤더 허용
)

# =====================================================
# 1. Fusion Model
# =====================================================

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

        # ✅ 이름: layers / norms1 / norms2 / ffns  (ckpt와 동일)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(nlayers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nlayers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nlayers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(nlayers)
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

        tokens = torch.stack([t0, t1, t2, t3, t4, t5], dim=1)  # (B,6,d)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)                    # (B,7,d)

        attn_last = None
        for attn, norm1, norm2, ffn in zip(self.layers, self.norms1, self.norms2, self.ffns):
            # ✅ head별 attn 유지
            attn_out, attn_weights = attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False
            )
            x = norm1(x + attn_out)
            x = norm2(x + ffn(x))
            attn_last = attn_weights  # (B, heads, 7, 7)

        cls_vec = x[:, 0]
        emo_logits = self.head_emo(cls_vec)
        arousal = self.head_arousal(cls_vec).squeeze(-1)
        valence = self.head_valence(cls_vec).squeeze(-1)

        if return_attn:
            cls_attn = attn_last.mean(1)[:, 0, 1:]  # (B,6)
            return emo_logits, arousal, valence, cls_attn

        return emo_logits, arousal, valence





# =====================================================
# 2. Load Models
# =====================================================


def normalize_fusion_state_dict(state_dict):
    """Map encoder-style checkpoints onto the current server model names."""
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

# ---------------------------
# Fusion 모델 가중치 경로
# ---------------------------
FUSION_MODEL_PATH = env_path("MUTON_FUSION_MODEL", "out/fusion_ko_final/final.pt")

pkg = torch.load(str(FUSION_MODEL_PATH), map_location="cpu")
cfg = pkg.get("backbone_cfg") or pkg.get("args", {})
model_state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
model_state = normalize_fusion_state_dict(model_state)
num_emotions = model_state["head_emo.weight"].shape[0]

fusion_model = FusionTransformer(
    d_model=cfg.get("d_model", 256),
    nhead=cfg.get("nhead", 8),
    nlayers=cfg.get("nlayers", 4),
    num_emotions=num_emotions
).to(DEVICE)

fusion_model.load_state_dict(model_state, strict=True)
fusion_model.eval()
print("Fusion model loaded")


# ---------------------------
# 텍스트 인코더(klue/roberta-small) 로드
# - get_text_embedding에서 mean pooling으로 문장 임베딩 구성
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")  # 토크나이저 로드
text_encoder = AutoModel.from_pretrained("klue/roberta-small").to(DEVICE).eval()  # 모델 로드 + eval

# ---------------------------
# 오디오/얼굴 인코더 로드
# - AudioEncoder: Whisper API + Silero VAD + WavLM features
# - FaceEncoder : JPEG → face_vec / emotion logits 등
# ---------------------------
audio_encoder = AudioEncoder()                                 # 오디오 인코더 인스턴스 생성
face_encoder = FaceEncoder()                                   # 얼굴 인코더 인스턴스 생성

# ---------------------------
# fusion output index → emotion label
# ---------------------------
if isinstance(pkg, dict) and pkg.get("ko_emo2id"):
    idx2emotion = {idx: label.capitalize() for label, idx in pkg["ko_emo2id"].items()}
else:
    idx2emotion = {
        0: "Angry",
        1: "Sad",
        2: "Disgust",
        3: "Surprise",
        4: "Happy",
        5: "Neutral",
        6: "Fear",
    }


# =====================================================
# Few-shot Exemplar Load (Emotion-conditioned)
# - GPT 요약 생성 시, 같은 감정 라벨의 예시 문장을 few-shot으로 넣어줌
# =====================================================
JSON_PATH = env_path("MUTON_MULTI_TEXT_JSON", "preprocessing/multi_text.json")
exemplar_dict = {}                                            # {Emotion(str): [summary(str), ...]}

try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:          # JSON 로드
        data = json.load(f)

    # data의 구조가 {something: [items]} 형태라고 가정하고 순회
    for _, items in data.items():                              # 각 그룹(items)을 순회
        for item in items:                                     # 아이템 하나씩 처리
            raw_emotion = item.get("emotion", "Neutral").capitalize()  # emotion 필드 추출 + capitalize
            if raw_emotion == "Dislike":                       # 데이터셋 라벨 불일치 보정
                raw_emotion = "Disgust"

            # JSON마다 summary 필드명이 다를 수 있어서 안전하게 후보 키를 순차 탐색
            summary = (
                item.get("summary_text")                       # 후보1
                or item.get("summary_textc")                   # 후보2
                or item.get("accessible_emotion_desc")         # 후보3
            )

            # summary가 있고 너무 짧지 않으면 exemplar_dict에 추가
            if summary and len(summary.strip()) > 5:
                exemplar_dict.setdefault(raw_emotion, []).append(summary.strip())

    # 로드 결과 로그 (각 감정별 개수)
    print(f"Few-shot exemplar loaded: { {k: len(v) for k,v in exemplar_dict.items()} }")

except Exception as e:
    # JSON 로드 실패 시 (경로 오류/형식 오류 등) → 빈 dict로 진행
    print("Failed to load exemplar_dict:", e)
    exemplar_dict = {}


# =====================================================
# Utils
# =====================================================

def get_text_embedding(text):
    # text: STT 결과 문자열
    # klue/roberta-small을 사용하여 문장 임베딩을 만듦 (mean pooling)

    inputs = tokenizer(
        text or "",                                           # None 대비
        return_tensors="pt",                                  # PyTorch tensor로 반환
        truncation=True,                                      # max_length 초과 시 자름
        padding="max_length",                                 # max_length로 패딩
        max_length=64                                         # 짧게 제한 (속도 + 안정)
    ).to(DEVICE)                                              # 입력 텐서를 DEVICE로 이동

    with torch.no_grad():                                     # 추론이므로 grad 비활성
        out = text_encoder(**inputs, return_dict=True)        # transformer forward
        mask = inputs["attention_mask"].unsqueeze(-1)         # (B, L, 1)
        # mean pooling: (hidden * mask).sum / mask.sum
        vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)

    return vec.cpu()                                          # CPU로 반환(이후 .to(DEVICE)로 다시 옮김)


def predict_fusion_emotion(face_out, feats, text):
    with torch.no_grad():
        batch = {
            "face_vec": torch.tensor(face_out["face_vec"], dtype=torch.float32).unsqueeze(0).to(DEVICE),
            "face_emo": torch.tensor(face_out["face_emotion_logits"], dtype=torch.float32).unsqueeze(0).to(DEVICE),
            "a_cont": torch.tensor(feats["content"], dtype=torch.float32).unsqueeze(0).to(DEVICE),
            "a_spk": torch.tensor(feats["speaker"], dtype=torch.float32).unsqueeze(0).to(DEVICE),
            "a_pros": torch.tensor(feats["prosody"], dtype=torch.float32).unsqueeze(0).to(DEVICE),
            "text": get_text_embedding(text).to(DEVICE),
        }

        logits, aro, val, cls_attn = fusion_model(batch, return_attn=True)
        probs = torch.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()

        emotion = idx2emotion[idx]
        conf = probs[0, idx].item()

        # cls_attn: (1,6) -> list[float]로 내보내기
        cls_attn = cls_attn[0].detach().cpu().numpy().tolist()

        return emotion, conf, float(aro.item()), float(val.item()), cls_attn




# =====================================================
# GPT Summary (🔥 temperature = attention 기반)
# - attention에서 text 비중이 높으면 temperature를 낮춰 더 보수적으로 요약
# - emotion에 맞는 few-shot 예시를 1~3개 넣어 문체를 고정
# =====================================================
def generate_situation_summary(emotion, conf, text, aro, val, cls_attn):
    # cls_attn: [face_vec, face_emo, audio_c, audio_s, audio_p, text]

    text_attn = float(cls_attn[-1])                               # 마지막 토큰이 text token이므로 text 주의도

    # attention 기반 temperature:
    # - text_attn이 클수록 더 deterministic하게(temperature 낮게)
    temperature = float(np.clip(0.6 - 0.3 * text_attn, 0.2, 0.6))

    # ---------------------------------
    # 같은 emotion의 few-shot 예시를 최대 3개 샘플링
    # ---------------------------------
    example_prompt = ""
    examples = exemplar_dict.get(emotion, [])

    if examples:                                                  # 예시가 존재하면
        shots = random.sample(examples, min(3, len(examples)))    # 최대 3개 랜덤 추출
        example_prompt = "\n".join([f"- {ex}" for ex in shots])   # bullet 형태로 합치기

    # ---------------------------------
    # GPT에게 주는 프롬프트
    # - 감정명 언급 금지 / 관찰 가능한 묘사만 / 종결어미 제한 등
    # - 40자 내외 한 문장
    # ---------------------------------
    prompt = f"""
역할: CCTV/바디캠 상황 요약 AI.

아래 [참고 예시]는 **'{emotion}' 상황에서 실제로 사용된 요약 문장**이다.
반드시 이 문체와 표현 방식을 따라,
현재 상황을 **한 문장으로** 묘사해라.

[참고 예시]
{example_prompt}

[현재 상황]
- 대사: "{text}"
- 각성도: {aro:.1f}
- 정서가: {val:.1f}

[작성 규칙]
1. 감정 이름 언급 금지
2. 설명, 분석, 판단 금지
3. 관찰 가능한 행동/말투/표정만 기술
4. 종결어미: "~함", "~중", "~보임"
5. 40자 내외, 한 문장

[결과]:
"""

    # ---------------------------------
    # OpenAI ChatCompletions 호출
    # - AudioEncoder 내부 client(OpenAI) 재사용
    # - temperature는 attention 기반
    # ---------------------------------
    res = audio_encoder.client.chat.completions.create(
        model="gpt-4o-mini",                                      # 빠르고 저렴한 모델
        messages=[{"role": "user", "content": prompt}],           # user 프롬프트로 전달
        temperature=temperature,                                  # attention 기반 temperature
        max_tokens=80                                             # 한 문장이라 짧게 제한
    )

    return res.choices[0].message.content.strip()                 # 출력 텍스트 반환


# =====================================================
# API
# =====================================================

@app.post("/process_video_chunk")
async def process_video_chunk(frame: UploadFile = File(...)):
    # Android에서 JPEG 프레임을 멀티파트로 전송하면
    # 여기서 bytes를 읽고 FaceEncoder로 얼굴 벡터/감정 추론 수행

    global latest_face_vec, face_vec_timestamp                    # 전역 캐시 갱신을 위해 global 선언

    jpeg = await frame.read()                                     # 업로드된 파일을 bytes로 읽기
    res = face_encoder.encode_jpeg_bytes(jpeg)                    # 얼굴 인코더 수행(구현은 face.py 내부)

    # face 인코딩이 성공하면 전역 캐시 업데이트
    if res.get("status") == "ok":
        latest_face_vec = res                                     # 최신 얼굴 결과 저장
        face_vec_timestamp = time.time()                          # 갱신 시각 저장

    return res                                                    # 얼굴 분석 결과(감정 등) 그대로 반환


@app.post("/process_audio_chunk")
async def process_audio_chunk(audio: UploadFile = File(...)):
    # Android에서 PCM chunk를 보내면
    # 1) AudioEncoder.stt_with_api로 STT (VAD/버퍼링 후 문장 단위로만 반환)
    # 2) 동일 PCM chunk로 WavLM 기반 feature 추출
    # 3) 결과를 JSON으로 반환 (Android가 2차 분석 트리거)

    pcm = await audio.read()                                      # 업로드된 pcm bytes 읽기

    # STT: audio.py 내부에서 VAD/버퍼 누적 후 조건 만족할 때만 텍스트 반환
    # 조건 미달이면 None → ""로 처리
    text = audio_encoder.stt_with_api(pcm) or ""

    # PCM bytes → float32 waveform ([-1, 1])로 변환
    pcm_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # WavLM 기반 특징 추출 (prosody/content/speaker 각각 768)
    feats = audio_encoder.extract_features_from_pcm(pcm_np)

    # Android에 빠르게 반환할 payload
    # - fusion_emotion/summary는 여기서는 비움 (2차 엔드포인트에서 수행)
    return {
        "text": text,                                             # STT 결과 (없으면 "")
        "prosody": feats["prosody"],                               # prosody vector (list[float], 768)
        "content": feats["content"],                               # content vector (list[float], 768)
        "speaker": feats["speaker"],                               # speaker vector (list[float], 768)
        "fusion_emotion": "",                                      # 2차에서 채움
        "summary": ""                                             # 2차에서 채움
    }


@app.post("/get_fusion_analysis")
async def get_fusion_analysis(
    text: str = Form(...),
    prosody: str = Form(...),
    content: str = Form(...),
    speaker: str = Form(...),
):
    global latest_face_vec, face_vec_timestamp

    if latest_face_vec is None or time.time() - face_vec_timestamp > 3.0:
        return {"fusion_emotion": "No Visual Input"}

    feats = {
        "prosody": json.loads(prosody),
        "content": json.loads(content),
        "speaker": json.loads(speaker),
    }

    # ✅ 여기서 fusion + attn까지
    emotion, conf, aro, val, cls_attn = predict_fusion_emotion(latest_face_vec, feats, text)

    summary = ""
    if conf > 0.4 and cls_attn is not None:
        summary = generate_situation_summary(emotion, conf, text, aro, val, cls_attn)

    return {
        "fusion_emotion": emotion,
        "fusion_confidence": conf,
        "arousal": aro,
        "valence": val,
        "summary": summary,
        "cls_attn": cls_attn,  # (선택) 디버깅/시각화용
    }



# =====================================================
# 서버 실행 엔트리포인트
# =====================================================
if __name__ == "__main__":
    # uvicorn으로 0.0.0.0:5000 바인딩 → 외부(클라우드플레어 터널 등) 접근 가능
    uvicorn.run(app, host="0.0.0.0", port=5000)
