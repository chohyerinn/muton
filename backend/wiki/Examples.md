# Examples

## Health Check

```bash
curl http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "status": "ok",
  "backend": "qwen_omni"
}
```

## Send One Video Frame

```bash
curl -X POST http://127.0.0.1:5000/process_video_chunk \
  -F "frame=@sample.jpg"
```

## Send One Audio Chunk

```bash
curl -X POST http://127.0.0.1:5000/process_audio_chunk \
  -F "audio=@chunk.pcm"
```

The response may contain an empty `text` field until the server detects the end of an utterance.

## Request Multimodal Summary

```bash
curl -X POST http://127.0.0.1:5000/get_fusion_analysis \
  -F "text=오늘 회의는 조금 급하게 진행되는 것 같아요." \
  -F "prosody=[]" \
  -F "content=[]" \
  -F "speaker=[]"
```

## Summarize A Conversation Record

```bash
curl -X POST http://127.0.0.1:5000/summarize_conversation_record \
  -H "Content-Type: application/json" \
  -d '{"conversation_text":"오늘 회의는 조금 급하게 진행되는 것 같아요.\n네, 핵심만 먼저 정리해 주세요."}'
```

Expected response:

```json
{
  "title": "회의 진행 상황을 빠르게 정리하는 대화"
}
```

## Python Client Example

See:

- `examples/python_api_client.py`

Example usage:

```bash
python examples/python_api_client.py \
  --base_url http://127.0.0.1:5000 \
  --image sample.jpg \
  --pcm chunk.pcm \
  --text "오늘 회의는 조금 급하게 진행되는 것 같아요."
```

## Cloudflare URL Check

After publishing a new tunnel URL:

```bash
curl "https://raw.githubusercontent.com/Ai-pre/MUTON/server_main/backend_url.json?t=$(date +%s)"
```

Then check the tunnel itself:

```bash
curl https://xxxxx.trycloudflare.com/health
```
