# Interview Listener — backend

FastAPI service that turns an interview question (+ optional resume) into a short
answer using the Groq API. The Android app and web frontend send the Groq key
**per request** in the `X-Groq-API-Key` header; this server never stores it.

This repository is **backend only** — the app/frontend lives elsewhere.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Readiness probe, returns `{"ok": true}` |
| `GET` | `/` | Service info |
| `POST` | `/answer` | Body `{ "question", "resume", "answer_style" }`, header `X-Groq-API-Key` |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deploy (Render)

`render.yaml` at the repo root defines the service. In Render: **New + → Blueprint**,
connect this repo, and it builds and starts automatically. Render provides HTTPS,
so the app can reach it from any network.

Env vars: `PYTHON_VERSION`, `GROQ_MODEL` (default `llama-3.1-8b-instant`), and
`ALLOWED_ORIGINS` (comma-separated CORS origins; default `*`).

> The free plan sleeps after ~15 minutes idle; the next request wakes it (~30–60s).
