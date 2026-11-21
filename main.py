import json, os, subprocess, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")  # any free-tier model works
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small")
MAX_MINUTES = int(os.environ.get("MAX_MINUTES", "90"))  # protects your server

WORK = Path("work"); WORK.mkdir(exist_ok=True)
OUT = Path("clips"); OUT.mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/clips", StaticFiles(directory=OUT), name="clips")

jobs: dict = {}
pool = ThreadPoolExecutor(max_workers=1)  # one video at a time keeps free hosting alive
_model = None


def whisper():
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_SIZE, device="auto", compute_type="int8")
    return _model


class Req(BaseModel):
    url: str
    max_clips: int = 5
    vertical: bool = True


def download(url: str, job_id: str) -> Path:
    opts = {
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(WORK / f"{job_id}.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if (info.get("duration") or 0) > MAX_MINUTES * 60:
            raise ValueError(f"Video is longer than {MAX_MINUTES} minutes.")
        ydl.download([url])
    return WORK / f"{job_id}.mp4"


def transcribe(path: Path):
    segments, _ = whisper().transcribe(str(path), vad_filter=True)
    return [(round(s.start, 1), round(s.end, 1), s.text.strip()) for s in segments]


PROMPT = """You are a short-form video editor who knows what goes viral on TikTok, Reels and Shorts.
Below is a timestamped transcript. Pick up to {n} clips of 20-60 seconds each.
Good clips: open with a strong hook in the first 3 seconds, hold one complete idea or story,
have emotion, surprise, a bold claim, or a useful tip, and end on a satisfying beat.
Clips must not overlap and must start and end on sentence boundaries.
Return ONLY a JSON list: [{{"start": seconds, "end": seconds, "title": "short title", "reason": "why it works", "score": 1-100}}]

Transcript:
{t}"""


def pick_clips(transcript, n):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set on the server.")
    text = "\n".join(f"[{a}-{b}] {t}" for a, b, t in transcript)
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_KEY},
        json={
            "contents": [{"parts": [{"text": PROMPT.format(n=n, t=text[:120000])}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=120,
    )
    r.raise_for_status()
    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    clips = json.loads(raw)
    clips = [c for c in clips if 5 < float(c["end"]) - float(c["start"]) <= 90]
    return sorted(clips, key=lambda c: -c.get("score", 0))[:n]


def cut(src: Path, c: dict, job_id: str, idx: int, vertical: bool) -> str:
    name = f"{job_id}_{idx}.mp4"
    vf = "crop=ih*9/16:ih,scale=720:1280" if vertical else "scale=-2:720"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(c["start"]), "-to", str(c["end"]), "-i", str(src),
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", "-movflags", "+faststart", str(OUT / name)],
        check=True, capture_output=True,
    )
    return name


def run(job_id: str, req: Req):
    job = jobs[job_id]
    src = None
    try:
        job["status"] = "Downloading video"
        src = download(req.url, job_id)
        job["status"] = "Listening to the video"
        transcript = transcribe(src)
        if not transcript:
            raise ValueError("No speech found in this video.")
        job["status"] = "Finding the best moments"
        picks = pick_clips(transcript, req.max_clips)
        job["status"] = "Cutting clips"
        for i, c in enumerate(picks):
            c["file"] = f"/clips/{cut(src, c, job_id, i, req.vertical)}"
        job["clips"] = picks
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        if src and src.exists():
            src.unlink()


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/api/clips")
def create(req: Req):
    job_id = uuid.uuid4().hex[:10]
    jobs[job_id] = {"status": "Queued", "clips": [], "error": None}
    pool.submit(run, job_id, req)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[job_id]
