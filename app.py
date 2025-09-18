# app.py
import os, glob, json
from typing import List, Optional
from datetime import datetime, timedelta
from dateutil.tz import tzlocal
import icalendar
import recurring_ical_events
from zoneinfo import ZoneInfo
import requests
import numpy as np
import faiss
import fitz  # PyMuPDF
from docx import Document
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import trafilatura
import io, base64
from PIL import Image
import mss
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"




# =========================
# Settings - Pre-sets
# =========================
LOCAL_TZ = ZoneInfo("Europe/Amsterdam")
APP_TITLE = "J.A.R.V.I.S BackEnd"
SUPPORTED_EXTS = (".txt", ".md", ".csv", ".pdf", ".docx")
INDEX_DIR = "./index"
META_PATH = os.path.join(INDEX_DIR, "meta.json")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")


# Models from Ollama
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.1:8b"
OLLAMA      = "http://localhost:11434"

# =========================
# FastAPI
# =========================
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ["http://127.0.0.1:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}
# Checking for Pulse


# =========================
# File parsing & chunking Functions
# =========================
def read_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md", ".csv"):
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext == ".pdf":
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    if ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def chunk_text(t: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    t = t.strip()
    if not t:
        return []
    chunks, i = [], 0
    while i < len(t):
        chunks.append(t[i:i + max_chars])
        i += max_chars - overlap
    return chunks

# =========================
# Embeddings & LLM
# =========================
def embed_one(text: str) -> np.ndarray:
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    vec = r.json()["embedding"]
    return np.array(vec, dtype="float32")

def embed_many(texts: List[str]) -> np.ndarray:
    # loooping
    return np.vstack([embed_one(t) for t in texts])

def llm_complete(prompt: str, temperature: float = 0.1) -> str:
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature}}
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

# =========================
# Index store (FAISS + meta)
# =========================
def ensure_store():
    os.makedirs(INDEX_DIR, exist_ok=True)
    if not os.path.exists(META_PATH):
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"chunks": [], "dim": None}, f)

def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_meta(meta):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)

def new_index(dim: int):
    # cosine similarity
    return faiss.IndexFlatIP(dim)

def save_index(index):
    faiss.write_index(index, INDEX_PATH)

def load_index():
    return faiss.read_index(INDEX_PATH)

# =========================
# /ingest  (parse → chunk → embed → index -< callouts )
# =========================
class IngestRequest(BaseModel):
    folder: str = "./Docs"
def as_local(dt):
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            # local for float
            return dt.replace(tzinfo=LOCAL_TZ).astimezone(LOCAL_TZ)
        return dt.astimezone(LOCAL_TZ)
    return dt

def event_to_text(ev, source_path: str) -> str:
    title = str(ev.get('summary', 'Untitled'))
    loc = str(ev.get('location', '') or '')
    desc = str(ev.get('description', '') or '')
    dtstart = as_local(ev['dtstart'].dt)
    dtend = as_local(ev.get('dtend', ev.get('dtstart')).dt)
    all_day = isinstance(ev['dtstart'].dt, datetime) is False or \
              (isinstance(ev['dtstart'].dt, datetime) and ev['dtstart'].dt.tzinfo is None and ev['dtstart'].dt.hour == 0 and ev['dtend'].dt.hour == 0)

    when_str = dtstart.strftime("%Y-%m-%d %H:%M") + " → " + dtend.strftime("%Y-%m-%d %H:%M")
    if all_day:
        when_str = dtstart.strftime("%Y-%m-%d") + (" (all day)" if dtend.date() == dtstart.date() else f" → {dtend.strftime('%Y-%m-%d')} (all day)")

    lines = [
        f"Event: {title}",
        f"When: {when_str}",
    ]
    if loc.strip():
        lines.append(f"Where: {loc}")
    if desc.strip():
        lines.append(f"Notes: {desc}")

    lines.append(f"Source: {os.path.basename(source_path)}")
    return "\n".join(lines)

def parse_ics_file(path: str, window_start: datetime, window_end: datetime):
    """Return expanded events between window_start and window_end."""
    with open(path, "rb") as f:
        cal = icalendar.Calendar.from_ical(f.read())

    events = recurring_ical_events.of(cal).between(window_start, window_end)
    out = []
    for ev in events:
        try:
            # double-checking files
            if 'dtstart' not in ev:
                continue
            out.append(ev)
        except Exception:
            continue
    return out

@app.post("/ingest")     # API Call on ingest function
def ingest(req: IngestRequest):
    ensure_store()

    # 1) collect files
    paths = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(req.folder, f"*{ext}")))
    paths = [p for p in paths if os.path.isfile(p)]

    # 2) parse + chunk
    all_chunks, meta_records = [], []
    for p in paths:
        text = read_text_from_file(p)
        for ch in chunk_text(text):
            meta_records.append({"path": p, "text": ch})
            all_chunks.append(ch)

    if not all_chunks:
        return {"added": 0, "files": len(paths), "message": "No valid chunks found (empty docs or unsupported types)."}

    # 3) embed
    vecs = embed_many(all_chunks).astype("float32")
    faiss.normalize_L2(vecs)

    # 4) build index
    index = new_index(vecs.shape[1])
    index.add(vecs)
    save_index(index)

    # 5) metadata
    save_meta({"chunks": meta_records, "dim": int(vecs.shape[1])})

    return {"added": len(all_chunks), "files": len(paths)}
class IngestURLRequest(BaseModel):
    url: str

@app.post("/ingest_url")     # API call on ingest webpages function
def ingest_url(req: IngestURLRequest):
    ensure_store()

    # 1) Download and extract text
    downloaded = trafilatura.fetch_url(req.url)
    if not downloaded:
        return {"added": 0, "message": "Could not fetch URL."}
    text = trafilatura.extract(downloaded)
    if not text:
        return {"added": 0, "message": "No readable text found at URL."}

    # 2) Chunk text
    chunks = chunk_text(text)
    if not chunks:
        return {"added": 0, "message": "No chunks created from URL text."}

    # 3) Embed
    vecs = embed_many(chunks).astype("float32")
    faiss.normalize_L2(vecs)

    # 4) Load or create index
    meta = load_meta()
    if os.path.exists(INDEX_PATH) and meta["dim"]:
        index = load_index()
    else:
        index = new_index(vecs.shape[1])
        meta["dim"] = int(vecs.shape[1])

    # 5) Add to index + metadata
    start_id = len(meta["chunks"])
    index.add(vecs)
    for ch in chunks:
        meta["chunks"].append({"path": req.url, "text": ch})

    save_index(index)
    save_meta(meta)

    return {"added": len(chunks), "source": req.url}


class IngestICSRequest(BaseModel):
    folder: str = "./Docs"
    days_ahead: int = 120            # how far to expand recurring events
    include_past_days: int = 7       # include recent past
    max_events: int = 1000           # safety cap

@app.post("/ingest_ics")     # API call to ingest calendar ( .ics ) files
def ingest_ics(req: IngestICSRequest):
    ensure_store()

    start = datetime.now(LOCAL_TZ) - timedelta(days=max(0, req.include_past_days))
    end   = datetime.now(LOCAL_TZ) + timedelta(days=max(1, req.days_ahead))

    # collect .ics files
    ics_paths = []
    for p in glob.glob(os.path.join(req.folder, "*.ics")):
        if os.path.isfile(p):
            ics_paths.append(p)

    if not ics_paths:
        return {"added": 0, "message": f"No .ics files found in {req.folder}."}

    # parse & flatten events
    texts = []
    for p in ics_paths:
        try:
            evs = parse_ics_file(p, start, end)
            for ev in evs:
                txt = event_to_text(ev, p)
                texts.append((p, txt))
        except Exception as e:
            # Skip broken calendars but keep going
            print(f"[ICS] Failed {p}: {e}")

    if not texts:
        return {"added": 0, "message": "No events found in window."}

    # Cap events for speed
    texts = texts[: max(1, req.max_events)]

    # Embed
    vecs = embed_many([t for _, t in texts]).astype("float32")
    faiss.normalize_L2(vecs)

    # Append to index + meta (create if needed)
    meta = load_meta()
    if os.path.exists(INDEX_PATH) and meta.get("dim"):
        index = load_index()
    else:
        index = new_index(vecs.shape[1])
        meta["dim"] = int(vecs.shape[1])

    index.add(vecs)
    for path, t in texts:
        meta["chunks"].append({"path": path, "text": t})

    save_index(index)
    save_meta(meta)

    return {"added": len(texts), "files": len(ics_paths), "window": {"start": start.isoformat(), "end": end.isoformat()}}

# =========================
# Retrieval helper
# =========================
def search_similar(query: str, top_k: int = 5):
    meta = load_meta()
    if not os.path.exists(INDEX_PATH) or not meta["chunks"]:
        return []
    index = load_index()
    q = embed_one(query).reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    sims, idxs = index.search(q, top_k)
    out = []
    for rank, i in enumerate(idxs[0].tolist()):
        if i < len(meta["chunks"]):
            c = meta["chunks"][i]
            out.append({
                "rank": rank + 1,
                "score": float(sims[0][rank]),
                "path": c["path"],
                "text": c["text"],
            })
    return out
# =========================
# Screen Review Support
# =========================

# HELPERS:
def grab_screenshot() -> Image.Image:
    with mss.mss() as sct:
        mon = sct.monitors[1]  # primary monitor
        raw = sct.grab(mon)
        return Image.frombytes("RGB", raw.size, raw.rgb)

def ocr_text(img: Image.Image) -> str:
    # light pre-processing improves OCR
    gray = img.convert("L")
    return pytesseract.image_to_string(gray)


# ENDPOINT - API CALL
class ScreenReviewRequest(BaseModel):
    goal: str = "Critique the design and code on screen; propose concrete fixes."
    top_k: int = 12
    target_words: int = 220

SCREEN_OCR_PROMPT = """You are JARVIS — an Experience Designer teammate.
SCREEN (OCR):
{ocr}

CONTEXT (design system, a11y, component APIs, style guides, heuristics):
{context}

Advise concretely:
1) Issues / risks (usability, clarity, a11y, code smells if visible)
2) Specific changes (labels, hierarchy, spacing, component choices, error handling, code snippets)
3) A 3-step plan to fix it now
Keep ≤{target_words} words. Cite with [source] where relevant.
"""

@app.post("/screen_review")
def screen_review(req: ScreenReviewRequest):
    img = grab_screenshot()
    text = ocr_text(img)

    # retrieve against OCR text + your goal
    query = (req.goal or "") + " " + text[:2000]
    hits = search_similar(query, top_k=max(5, min(50, req.top_k))) or []
    ctx = "\n\n".join(f"[{os.path.basename(h['path'])}]\n{h['text']}" for h in hits)

    prompt = SCREEN_OCR_PROMPT.format(ocr=text[:5000], context=ctx, target_words=req.target_words)
    answer = llm_complete(prompt, temperature=0.3)
    sources = list({h["path"] for h in hits})
    return {"answer": answer, "sources": sources, "ocr_preview": text[:400]}







# =========================
# /ask   # Needed only on backend page / Is called when press "Ask" on react page
# =========================
class AskRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/ask")
def ask(req: AskRequest):
    hits = search_similar(req.question, top_k=max(1, min(20, req.top_k)))
    if not hits:
        return {"answer": "Index is empty. Run /ingest first.", "sources": []}

    ctx = "\n\n".join(f"[{os.path.basename(h['path'])}]\n{h['text']}" for h in hits)
    prompt = f"""Answer using ONLY the CONTEXT. Cite with [filename.ext].
If information is missing, say so briefly.

QUESTION:
{req.question}

CONTEXT:
{ctx}
"""
    answer = llm_complete(prompt)
    seen, sources = set(), []
    for h in hits:
        p = h["path"]
        if p not in seen:
            seen.add(p); sources.append(p)
    return {"answer": answer, "sources": sources}

# =========================
# /synthesize  (Summarisation of a document)
# =========================
class SynthesizeRequest(BaseModel):
    query: Optional[str] = None   # topic; if None → global sample
    top_k: int = 30               # number of chunks considered
    target_words: int = 150       # length of final brief

MAP_PROMPT = """Extract the 2–3 most important facts from the source chunk below.
- Tight bullet points
- Quote key numbers/dates exactly
- No speculation

<<<
{chunk}
>>>"""

REDUCE_PROMPT = """Write a concise synthesis in ≤{target_words} words.
- Clear prose (no bullets)
- Add bracketed citations like [filename.ext]
- If evidence is thin or conflicting, say so briefly

NOTES:
{notes}
"""

@app.post("/synthesize")
def synth(req: SynthesizeRequest):
    meta = load_meta()
    if not meta["chunks"]:
        return {"answer": "Index is empty. Run /ingest first.", "sources": []}

    # choose chunks: either top_k by query or first N for global
    if req.query:
        hits = search_similar(req.query, top_k=max(5, min(200, req.top_k)))
    else:
        n = max(5, min(200, req.top_k))
        hits = [{"path": c["path"], "text": c["text"]} for c in meta["chunks"][:n]]

    # MAP
    notes_blocks, sources = [], []
    for h in hits:
        fname = os.path.basename(h["path"])
        bullets = llm_complete(MAP_PROMPT.format(chunk=h["text"][:5000]), temperature=0.0)
        if bullets.strip():
            notes_blocks.append(f"[{fname}]\n{bullets}")
            sources.append(h["path"])

    if not notes_blocks:
        return {"answer": "No salient notes extracted.", "sources": []}

    # REDUCE
    notes = "\n\n".join(notes_blocks[:200])
    answer = llm_complete(REDUCE_PROMPT.format(target_words=req.target_words, notes=notes), temperature=0.2)

    # dedupe sources
    out_sources, seen = [], set()
    for s in sources:
        if s not in seen:
            seen.add(s); out_sources.append(s)

    return {"answer": answer, "sources": out_sources[:20]}
