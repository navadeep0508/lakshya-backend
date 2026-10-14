import os
import httpx
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Interview Listener API", version="1.0.0")

# Comma-separated list of allowed origins (set ALLOWED_ORIGINS in your host's
# dashboard to lock this down). Defaults to "*" which is fine for local dev and
# for the Android/desktop app, which sends no browser Origin you can rely on.
_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=6000)
    resume: str = Field(default="", max_length=30000)
    answer_style: str = Field(default="Speakable: concise and natural", max_length=120)

@app.get("/")
def root():
    return {"service": "Interview Listener API", "docs": "/docs", "health": "/health", "answer": "POST /answer", "transcribe": "POST /transcribe"}

@app.get("/health")
def health():
    return {"ok": True, "service": "Interview Listener API"}

@app.post("/answer")
async def answer(payload: AnswerRequest, x_groq_api_key: str = Header(default="")):
    api_key = x_groq_api_key.strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="Add your Groq API key in the app.")
    if len(payload.question.strip()) < 3:
        raise HTTPException(status_code=422, detail="Please provide a complete question.")
    style = payload.answer_style
    system = (
        "You are an interview practice coach. Write an accurate answer the candidate can say naturally. "
        "Use plain, confident English and answer the actual question directly. Usually use 2-5 sentences. "
        "Use resume details only when relevant; never invent personal experience, employers, metrics, or projects. "
        "If the question asks about experience not in the resume, give a truthful general answer or say what the candidate should customize. "
        "For coding/DSA, state the approach, then code only if requested, and include time/space complexity when useful. "
        "For behavioral questions, use STAR concisely and ground claims in provided facts. Avoid prefacing with 'Here is your answer'."
    )
    user = f"Answer style: {style}\n\nRESUME / PROJECT CONTEXT:\n{payload.resume[:30000] or '[No resume supplied]'}\n\nINTERVIEW QUESTION:\n{payload.question.strip()}"
    try:
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.3,
            "max_tokens": 450,
        }
        # GPT-OSS models expose a reasoning budget; keep it low so answers stay
        # fast and spoken-style rather than thinking out loud.
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
            )
        if resp.status_code >= 400:
            detail = "Groq rejected the request. Check your API key, model availability, and rate limits."
            if resp.status_code in (401, 403): detail = "Invalid or unauthorized Groq API key."
            elif resp.status_code == 429: detail = "Groq rate limit reached. Wait a moment and try again."
            raise HTTPException(status_code=resp.status_code if resp.status_code in (401, 403, 429) else 502, detail=detail)
        data = resp.json()
        return {"answer": data["choices"][0]["message"]["content"].strip()}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI request timed out. Try again.")
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Could not connect to Groq. Check the backend internet connection.")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), x_groq_api_key: str = Header(default="")):
    """Speech-to-text via Groq Whisper.

    The Android app records audio with AudioRecord (the device's own cloud
    speech service is unreliable/absent on many phones) and posts the WAV here.
    Groq's Whisper is fast and works with the same API key used for answers.
    """
    api_key = x_groq_api_key.strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="Add your Groq API key in the app.")
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=422, detail="No audio received.")
    # Guard against runaway recordings (roughly 25 MB ~ 13 min of 16 kHz 16-bit mono).
    if len(audio) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio clip is too long.")
    filename = (file.filename or "speech.wav").lower()
    content_type = file.content_type or "audio/wav"
    if not (filename.endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac")) or (content_type or "").startswith("audio/")):
        raise HTTPException(status_code=415, detail="Please send a WAV, MP3, M4A, or OGG audio clip.")
    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (filename or "speech.wav", audio, content_type or "audio/wav")},
                data={
                    "model": os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
                    "response_format": "json",
                    "temperature": "0",
                },
            )
        if resp.status_code >= 400:
            detail = "Groq rejected the audio. Check your API key and rate limits."
            if resp.status_code in (401, 403): detail = "Invalid or unauthorized Groq API key."
            elif resp.status_code == 429: detail = "Groq rate limit reached. Wait a moment and try again."
            raise HTTPException(status_code=resp.status_code if resp.status_code in (401, 403, 429) else 502, detail=detail)
        data = resp.json()
        return {"text": (data.get("text") or "").strip()}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Transcription timed out. Try again.")
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Could not connect to Groq. Check the backend internet connection.")
