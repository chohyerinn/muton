# MUTON

> 농인·난청인을 위한 실시간 대화 보조 시스템 — 단순 자막을 넘어 **표정·음성 맥락·대화 흐름**까지 요약해 상황과 감정의 뉘앙스를 전달합니다.

가천대학교 졸업 팀 프로젝트 · Python 71% / Kotlin 29%

> **이 저장소는 팀 프로젝트입니다.**
> AI 모델 학습·백엔드 서버 로직은 팀원이 맡았으며, 아래에서 제 기여 범위는 Android 클라이언트 전반 — UI, Firebase 인증(회원가입/로그인), 백엔드 API 연동 입니다.
>
> 원본: [Ai-pre/MUTON](https://github.com/Ai-pre/MUTON) · Android 원본: [Ai-pre/MUTON-Android](https://github.com/Ai-pre/MUTON-Android)
> 본 레포는 분리되어 있던 백엔드/AI·Android 레포를 하나로 묶은 포트폴리오용 통합 정리본입니다.

---

## 데모

| 실시간 자막 | 대화 요약 |
|---|---|
| *(스크린샷 또는 GIF 추가)* | *(스크린샷 또는 GIF 추가)* |

> TODO: 시연 GIF 1~2개를 넣으세요. 채용자는 README에서 "동작하는 화면"을 가장 먼저 봅니다.

---

## 풀려는 문제

기존 음성→자막 도구는 **"무슨 말을 했는지"**는 알려주지만 **"어떤 상황·감정이었는지"**는 놓칩니다. 농인·난청인은 화자의 표정, 말의 톤, 대화의 맥락을 종합한 정보가 필요합니다.

MUTON은 카메라·마이크 입력을 받아:
1. **음성 → 텍스트** (Whisper STT)
2. **표정 + 음성 맥락 + 대화 흐름**을 멀티모달로 요약 (Qwen2.5-Omni + LoRA)
3. Android 화면에 **실시간 자막과 상황 요약**을 함께 표시

합니다.

---

## 시스템 구조

```
[Android 클라이언트]                 [FastAPI 백엔드]
 카메라/오디오 캡처  ──스트리밍──▶   Whisper STT
 실시간 자막 표시    ◀──결과────    Qwen2.5-Omni (LoRA) 멀티모달 요약
 요약 화면                          ROUGE-L / 지연시간 평가
```

- `android/` — Kotlin 클라이언트 (캡처, 실시간 자막, 요약 화면)
- `backend/` — FastAPI 서버, STT, 멀티모달 처리, 학습 스크립트
- `docs/` — 설계 노트, 포트폴리오 문서

---

## 내가 담당한 부분 (Android 클라이언트 전반)

- **UI/화면 설계 및 구현**: 실시간 자막 화면 ↔ 대화 요약 화면 전환 등, 사용 시나리오에 맞춘 화면 흐름 전체
- **Firebase 인증**: 회원가입·로그인 등 사용자 인증/관리 기능 구현 (Firebase Auth) — 클라이언트에서 사용자 상태를 관리하고 인증 흐름을 처리
- **백엔드 API 연동**: FastAPI 엔드포인트와의 요청/응답 연동 구현. 백엔드 처리 단계(STT → 멀티모달 요약)의 흐름을 이해하고, 응답을 화면 표시 타이밍에 맞춰 렌더링
- **실시간 캡처 파이프라인**: 카메라/오디오 입력을 캡처해 백엔드로 전송하고, 결과를 끊김 없이 자막으로 갱신

> AI 모델 학습·서버 로직은 직접 작성하지 않았지만, **인증·API 연동·실시간 표시**까지 클라이언트–백엔드 경계를 책임지며 전체 시스템 흐름을 이해하고 있습니다.

> Firebase 설정 파일(`google-services.json` 등)은 보안상 포트폴리오 아카이브에서 제외했습니다.

---

## 기술 스택

| 영역 | 기술 |
|---|---|
| Android | Kotlin, Camera/Audio 캡처, 실시간 자막 렌더링 |
| 백엔드 | FastAPI |
| STT | OpenAI Whisper |
| 멀티모달 요약 | Qwen2.5-Omni + LoRA 파인튜닝 |
| 평가 | ROUGE-L, 지연시간 측정, 사람 평가 |

---

## 평가 (팀 성과)

실시간성과 출력 품질을 함께 봤습니다:
- **요약 품질**: ROUGE-L 기반 정량 평가 + 출력에 대한 사람 평가
- **실시간성**: 입력→자막 표시까지의 지연시간 측정

> TODO: 가능하면 실제 수치(ROUGE-L 점수, 평균 지연 ms)를 한 줄 넣으세요. 숫자 하나가 README의 신뢰도를 크게 올립니다.

---

## 회고

> TODO(직접 작성 권장): Android 담당으로서 가장 어려웠던 점(예: 실시간 스트리밍 지연/끊김 처리),
> 백엔드와 계약을 맞추며 배운 점, 다시 한다면 바꿀 점을 2~3줄로. 면접에서 그대로 답변 소재가 됩니다.

---

## 실행 방법

### 백엔드

```bash
cd backend
pip install -r requirements.txt
pip install -r requirements-qwen-omni.txt
python scripts/run_qwen_server.py
```

모델 자산, LoRA 어댑터 경로, API 키는 로컬에서 별도 설정이 필요합니다. 자세한 원본 설정은 [backend/README.md](backend/README.md) 참고.

### Android 클라이언트

`android/`를 Android Studio에서 열거나 커맨드라인으로 빌드:

```powershell
cd android
.\gradlew.bat :app:assembleDebug
```

Firebase 기반 로그인/기록 기능을 쓰려면 `android/app/google-services.example.json`을 `android/app/google-services.json`으로 복사한 뒤 로컬 Firebase 프로젝트 설정을 채우세요. 실제 설정 파일은 보안상 이 아카이브에서 제외(.gitignore)되어 있습니다.

---

## 참고

- 본 레포는 학내 팀 프로젝트의 포트폴리오 아카이브이며, 별도 재배포본이 아닙니다.
- 런타임 URL, 모델 경로, API 키, 터널 설정은 라이브 데모 전에 각자 구성해야 합니다.
