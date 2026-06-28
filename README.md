# MUTON

> 가천대학교 졸업작품 팀 프로젝트. 원본: https://github.com/Ai-pre/MUTON  
> Android 원본: https://github.com/Ai-pre/MUTON-Android  
> 본 레포는 포트폴리오용 통합 정리본입니다.

MUTON은 청각장애인과 난청 사용자를 위한 실시간 옴니모달 대화 보조 시스템입니다. 음성 인식 결과에 얼굴 표정, 음성 맥락, 대화 흐름을 함께 반영해 단순 자막을 넘어 대화의 감정과 상황을 전달하는 것을 목표로 했습니다.

이 레포는 원래 분리되어 있던 백엔드/AI 레포와 Android 클라이언트 레포를 하나의 구조로 묶은 archive입니다.

## Repository Layout

```text
muton/
  backend/   FastAPI server, STT, multimodal processing, model training/evaluation scripts
  android/   Android client for camera/audio capture, subtitle display, and summary screens
  docs/      source commit and portfolio notes for this integrated archive
```

## System Overview

| Area | Implementation |
|---|---|
| Client | Android app for camera/audio capture and live result display |
| Backend | FastAPI server exposed through a tunnel during demos |
| STT | OpenAI Whisper-based transcription path with local fallback history |
| Multimodal summary | Qwen2.5-Omni with LoRA adaptation |
| Evaluation | ROUGE-L comparison, human evaluation, input ablation, latency checks |


## Run Backend

```bash
cd backend
pip install -r requirements.txt
pip install -r requirements-qwen-omni.txt
python scripts/run_qwen_server.py
```

The backend expects model assets, LoRA adapter paths, and API keys to be configured locally. See [backend/README.md](backend/README.md) for the original detailed setup.

## Run Android Client

Open `android/` in Android Studio, or build from the command line:

```powershell
cd android
.\gradlew.bat :app:assembleDebug
```

For Firebase-backed login or record features, copy `android/app/google-services.example.json` to `android/app/google-services.json` and fill in a local Firebase project configuration. The real config file is intentionally ignored in this portfolio archive.


## Notes

- This repository is a portfolio archive of an academic team project, not a separately relicensed distribution.
- Runtime URLs, model paths, API keys, and tunnel settings must be configured separately before running a live demo.
