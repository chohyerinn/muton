# MUTON

> 가천대학교 졸업작품 팀 프로젝트. 원본: https://github.com/Ai-pre/MUTON  
> Android 원본: https://github.com/Ai-pre/MUTON-Android  
> 본 레포는 포트폴리오용 통합 정리본이며, 팀 프로젝트 전체 결과와 제가 담당한 부분을 구분해 명시합니다.

MUTON은 청각장애인과 난청 사용자를 위한 실시간 옴니모달 대화 보조 시스템입니다. 음성 인식 결과에 얼굴 표정, 음성 맥락, 대화 흐름을 함께 반영해 단순 자막을 넘어 대화의 감정과 상황을 전달하는 것을 목표로 했습니다.

이 레포는 원래 분리되어 있던 백엔드/AI 레포와 Android 클라이언트 레포를 포트폴리오 검토가 쉽도록 하나의 구조로 묶은 archive입니다.

## Repository Layout

```text
muton/
  backend/   FastAPI server, STT, multimodal processing, model training/evaluation scripts
  android/   Android client for camera/audio capture, subtitle display, and summary screens
  docs/      source commit and portfolio notes for this integrated archive
```

## My Contributions

- 멀티모달 입력 흐름 구성: 음성, 얼굴 프레임, 텍스트 대화 맥락을 요약 모델 입력으로 연결
- LoRA 기반 2단계 학습 파이프라인 정리: MELD 기반 pseudo-summary stage와 한국어 데이터 적응 stage 구성
- FastAPI 추론 서버 구현 및 Android 연동 API 구성
- Android 실시간 데모 흐름 개선: 백엔드 URL 로딩, 자막, 감정, 요약 결과 표시 흐름 연결
- 평가 문서화: Base vs LoRA 비교, 입력 ablation, human evaluation, latency 결과 정리

## System Overview

| Area | Implementation |
|---|---|
| Client | Android app for camera/audio capture and live result display |
| Backend | FastAPI server exposed through a tunnel during demos |
| STT | OpenAI Whisper-based transcription path with local fallback history |
| Multimodal summary | Qwen2.5-Omni with LoRA adaptation |
| Evaluation | ROUGE-L comparison, human evaluation, input ablation, latency checks |

## Original Sources

This archive was created from the following upstream commits:

| Component | Upstream | Branch | Commit |
|---|---|---|---|
| Backend / AI | https://github.com/Ai-pre/MUTON | `server_main` | `b79a68efa824ec478e7f7431c4c1667c05bc9102` |
| Android | https://github.com/Ai-pre/MUTON-Android | `main` | `e0a7cb7511c34a9a5ce857e496a3ed0e9ce1d45c` |

See [docs/SOURCE_COMMITS.md](docs/SOURCE_COMMITS.md) for the archive notes.

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

See [android/README.md](android/README.md) for the original client documentation.

## Notes

- This repository is a portfolio archive of an academic team project, not a separately relicensed distribution.
- Runtime URLs, model paths, API keys, and tunnel settings must be configured separately before running a live demo.
- The original project documentation is preserved under `backend/README.md`, `backend/docs/`, `backend/wiki/`, and `android/README.md`.

## License / Reuse

The upstream repositories were shared for academic review and open-source release preparation, but did not include a formal license file at the time this archive was created. Do not reuse, redistribute, or commercialize the code without checking the original project authors' permission and current upstream license state.

See [LICENSE_NOTICE.md](LICENSE_NOTICE.md).
