# MUTON-Android

MUTON-Android is the Android client for MUTON, a real-time omnimodal dialogue assistance system for hearing-impaired users. The app captures camera frames and microphone audio, streams them to the MUTON backend, and displays subtitles, visual emotion cues, and image/audio/text-based summaries in a mobile interface.

## Contents

- [Overview](#overview)
- [App Features](#app-features)
- [Installation](#installation)
- [Backend Connection](#backend-connection)
- [Running On Device](#running-on-device)
- [Screens](#screens)
- [Related Repository](#related-repository)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

The Android app is the user-facing part of MUTON. While the backend handles STT, face/audio processing, and Qwen2.5-Omni based summary generation, this client focuses on real-time capture, request synchronization, and presenting the result in a practical conversation flow.

In Graduation Project 2, the app was refined for a more stable live demo. It now discovers the active backend address dynamically, keeps the UI compatible with backend model changes, and routes conversation record summaries through the server so API keys are not stored inside the APK.

## App Features

- Camera frame capture for visual context
- Microphone audio capture for streaming STT
- Real-time requests to the MUTON backend
- Subtitle display from finalized speech utterances
- Visual emotion display from face-frame analysis
- Multimodal summary display from backend fusion analysis
- Conversation record screens and server-side record summary requests
- Dynamic backend URL loading from the main MUTON repository

## Installation

Open the project in Android Studio.

Recommended project configuration:

- `compileSdk`: 35
- `minSdk`: 28
- `targetSdk`: 35
- package namespace: `com.example.myapplication`

Build the app module from Android Studio, or use Gradle from the project root:

```bash
./gradlew :app:assembleDebug
```

On Windows PowerShell:

```powershell
.\gradlew.bat :app:assembleDebug
```

## Backend Connection

The app reads the active backend URL from the main MUTON repository:

```text
https://api.github.com/repos/Ai-pre/MUTON/contents/backend_url.json?ref=server_main
```

The backend URL file points to the current Cloudflare Tunnel address. This keeps the Android app stable even when the server tunnel changes between demos.

Runtime relationship:

- Android sends audio chunks to `/process_audio_chunk`.
- Android sends camera frames to `/process_video_chunk`.
- Android requests multimodal summaries from `/get_fusion_analysis`.
- Android requests conversation record summaries from `/summarize_conversation_record`.

## Running On Device

Before launching the app, make sure the backend server is running and the current Cloudflare Tunnel URL has been published to `backend_url.json`.

The device or emulator must allow:

- Camera access
- Microphone access
- Network access

For the live demo, a physical Android device is recommended because camera, microphone, and network timing are closer to the intended use case.

## Screens

![MUTON Android main screen](https://github.com/user-attachments/assets/67b02611-b136-4166-8215-c3986d915b1e)

![MUTON Android live screen](https://github.com/user-attachments/assets/ec7bd87a-aca3-4b09-bdc4-7fe2890d4703)

## Related Repository

Backend runtime, API implementation, model training scripts, dataset processing, and wiki documentation are maintained in the main MUTON repository:

- [Ai-pre/MUTON](https://github.com/Ai-pre/MUTON)

The main repository includes the project poster, system pipeline, API documentation, evaluation summary, and training details.

## Project Structure

```text
MUTON-Android/
  app/
    src/main/java/com/example/myapplication/
      MainActivity.kt
      OpenAiSummaryService.kt          server summary endpoint client; no embedded API key
      ConversationRecordStore.kt
      RecordDetailActivity.kt
      HomeActivity.kt
      SettingsActivity.kt
    src/main/res/
  gradle/
  build.gradle.kts
  settings.gradle.kts
```

## License

This repository is currently shared for academic review and open-source release preparation. Add a formal license file before external reuse, redistribution, or commercial use.
