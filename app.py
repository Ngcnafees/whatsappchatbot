# app1.py
import os, json, hmac, hashlib, traceback, re, mimetypes
from urllib.parse import urlparse
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
from io import BytesIO

import httpx
import pandas as pd
import numpy as np
import faiss
from PIL import Image

import torch
import open_clip

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from qa import ask_docs, ask_openai
from salesforce_products import sf_product_search

load_dotenv()

# =========================================================
# WhatsApp Cloud API ENV
# =========================================================
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "").strip()
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "").strip()
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID", "").strip()
META_APP_SECRET = os.getenv("META_APP_SECRET", "").strip()
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v24.0").strip()

SAFE_FALLBACK_MESSAGE = os.getenv(
    "SAFE_FALLBACK_MESSAGE",
    "Sorry, I’m having trouble right now. Please try again later.",
).strip()

DEBUG_SF = os.getenv("DEBUG_SF", "false").lower() == "true"
SKIP_SIGNATURE_VERIFY = os.getenv("SKIP_SIGNATURE_VERIFY", "false").lower() == "true"

# =========================================================
# Excel mapping ENV
# =========================================================
IMAGE_XLSX_PATH = os.getenv("IMAGE_XLSX_PATH", "data/Full_stock_Cleaned.xlsx").strip()
EXCEL_STOCK_COL = os.getenv("EXCEL_STOCK_COL", "stock_code").strip()
EXCEL_URL_COL = os.getenv("EXCEL_URL_COL", "image_url").strip()

# =========================================================
# Similarity index ENV
# =========================================================
IMAGE_INDEX_DIR = os.getenv("IMAGE_INDEX_DIR", "image_vectorstore").strip()
SIM_TOP_K = int(os.getenv("SIM_TOP_K", "3"))

CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32").strip()
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k").strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Globals
# =========================================================
_stock_to_url_norm: Dict[str, str] = {}
_stock_to_media_id: Dict[str, str] = {}
_seen: Dict[str, int] = {}

_img_index = None
_img_meta: List[dict] = []

_clip_model = None
_clip_preprocess = None

# =========================================================
# Command patterns (accept ANYTHING after)
# =========================================================
IMG_PATTERN = re.compile(r"^\s*IMG-\s*(.+?)\s*$", re.IGNORECASE)
SIM_PATTERN = re.compile(
    r"^\s*SIM(?:ILAR)?-\s*(.+?)\s*$|^\s*SIM(?:ILAR)?\s+(.+?)\s*$",
    re.IGNORECASE,
)

def extract_img_raw(text: str) -> Optional[str]:
    m = IMG_PATTERN.match(text or "")
    return m.group(1) if m else None

def extract_sim_raw(text: str) -> Optional[str]:
    m = SIM_PATTERN.match(text or "")
    if not m:
        return None
    return m.group(1) or m.group(2)

# =========================================================
# Normalization for lookup (Excel/SF/cache)
# =========================================================
def norm_lookup(code: str) -> str:
    if code is None:
        return ""
    s = code.strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    return s

def safe_log_exc(prefix: str):
    print(prefix)
    traceback.print_exc()

def require_meta_env():
    if not META_ACCESS_TOKEN or not META_PHONE_NUMBER_ID:
        raise RuntimeError("META_ACCESS_TOKEN / META_PHONE_NUMBER_ID not set")

# =========================================================
# Excel map load
# =========================================================
def load_excel_map() -> int:
    global _stock_to_url_norm

    if not os.path.exists(IMAGE_XLSX_PATH):
        print(f"[WARN] Excel not found: {IMAGE_XLSX_PATH}")
        _stock_to_url_norm = {}
        return 0

    df = pd.read_excel(IMAGE_XLSX_PATH)
    if EXCEL_STOCK_COL not in df.columns or EXCEL_URL_COL not in df.columns:
        raise RuntimeError(
            f"Excel must contain columns: {EXCEL_STOCK_COL}, {EXCEL_URL_COL}. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=[EXCEL_STOCK_COL, EXCEL_URL_COL]).copy()
    df[EXCEL_STOCK_COL] = df[EXCEL_STOCK_COL].astype(str)
    df[EXCEL_URL_COL] = df[EXCEL_URL_COL].astype(str).str.strip()

    tmp: Dict[str, str] = {}
    for _, row in df.iterrows():
        raw_code = str(row[EXCEL_STOCK_COL])
        url = str(row[EXCEL_URL_COL]).strip()
        if not raw_code or not url:
            continue
        key = norm_lookup(raw_code)
        if key and key not in tmp:
            tmp[key] = url

    _stock_to_url_norm = tmp
    print(f"[OK] Loaded {len(_stock_to_url_norm)} image mappings from Excel.")
    return len(_stock_to_url_norm)

def get_image_url_for_code(code_any: str) -> Optional[str]:
    if not _stock_to_url_norm:
        try:
            load_excel_map()
        except Exception as e:
            print("[ERROR] Excel load failed:", e)
            return None
    return _stock_to_url_norm.get(norm_lookup(code_any))

# =========================================================
# Signature verify
# =========================================================
def verify_signature(body: bytes, header: Optional[str]):
    if SKIP_SIGNATURE_VERIFY:
        return
    if not META_APP_SECRET:
        return
    if not header or not header.startswith("sha256="):
        raise HTTPException(status_code=403, detail="Bad signature header")

    expected = hmac.new(META_APP_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    received = header.split("sha256=", 1)[1]
    if not hmac.compare_digest(expected, received):
        raise HTTPException(status_code=403, detail="Bad signature")

# =========================================================
# WhatsApp payload helpers
# =========================================================
def extract_messages(payload: dict) -> List[dict]:
    msgs: List[dict] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value") or {}
            for msg in (value.get("messages") or []):
                msgs.append(msg)
    return msgs

def get_sender(msg: dict) -> str:
    return (msg.get("from") or "").strip()

def get_msg_id(msg: dict) -> str:
    return (msg.get("id") or "").strip()

def extract_text(msg: dict) -> str:
    if msg.get("type") == "text":
        return (msg.get("text") or {}).get("body", "")
    return ""

def extract_image_media_id(msg: dict) -> Optional[str]:
    if msg.get("type") == "image":
        return (msg.get("image") or {}).get("id")
    return None

# =========================================================
# Load FAISS index + meta
# =========================================================
def load_image_similarity_index() -> int:
    global _img_index, _img_meta

    idx_path = os.path.join(IMAGE_INDEX_DIR, "index.faiss")
    meta_path = os.path.join(IMAGE_INDEX_DIR, "meta.jsonl")

    print(f"[SIM] Loading index: {idx_path}")
    print(f"[SIM] Loading meta : {meta_path}")

    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        print("[SIM][WARN] Missing index.faiss or meta.jsonl")
        _img_index = None
        _img_meta = []
        return 0

    _img_index = faiss.read_index(idx_path)

    _img_meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                _img_meta.append(json.loads(line))

    print(f"[SIM] FAISS loaded: ntotal={_img_index.ntotal} meta_rows={len(_img_meta)}")
    return len(_img_meta)

# =========================================================
# OpenCLIP load + embed
# =========================================================
def load_clip_model():
    global _clip_model, _clip_preprocess
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    _clip_model = _clip_model.to(DEVICE).eval()
    print(f"[SIM] OpenCLIP loaded: {CLIP_MODEL}/{CLIP_PRETRAINED} on {DEVICE}")

@torch.no_grad()
def embed_image_bytes(img_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    x = _clip_preprocess(img).unsqueeze(0).to(DEVICE)
    v = _clip_model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.squeeze(0).detach().cpu().numpy().astype("float32")

def faiss_search(vec: np.ndarray, k: int) -> List[dict]:
    if _img_index is None or not _img_meta:
        return []
    q = vec.reshape(1, -1).astype("float32")
    scores, ids = _img_index.search(q, k + 6)

    out: List[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(_img_meta):
            continue
        row = dict(_img_meta[idx])
        row["score"] = float(score)
        out.append(row)
        if len(out) >= k:
            break
    return out

# =========================================================
# WhatsApp send/download helpers
# =========================================================
async def wa_send_text(to: str, text: str):
    require_meta_env()
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text[:4096]}}
    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            print("Meta send text error:", resp.status_code, resp.text)

async def wa_upload_image_from_url(image_url: str) -> str:
    require_meta_env()
    async with httpx.AsyncClient(timeout=60) as client:
        img_resp = await client.get(image_url)
        img_resp.raise_for_status()
        img_bytes = img_resp.content

    path = urlparse(image_url).path
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{META_PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
    files = {"file": ("image", img_bytes, mime)}
    data = {"messaging_product": "whatsapp"}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, data=data, files=files)
        resp.raise_for_status()
        return resp.json()["id"]

async def wa_send_image_id(to: str, media_id: str, caption: str = ""):
    require_meta_env()
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": {"id": media_id, **({"caption": caption[:1024]} if caption else {})},
    }
    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            print("Meta send image error:", resp.status_code, resp.text)

async def wa_download_media_bytes(media_id: str) -> bytes:
    require_meta_env()
    meta_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{media_id}"
    headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(meta_url, headers=headers)
        r.raise_for_status()
        data = r.json()

    dl_url = data.get("url")
    if not dl_url:
        raise RuntimeError(f"No download url in media response: {data}")

    async with httpx.AsyncClient(timeout=60) as client:
        r2 = await client.get(dl_url, headers=headers)
        r2.raise_for_status()
        return r2.content

# =========================================================
# Salesforce mapping (supports your NEW fields)
# =========================================================
def pick_qty(row: dict) -> str:
    v = row.get("quantity")
    if v is None:
        v = row.get("qoh")
    if v is None:
        v = 0
    try:
        if isinstance(v, (int, float)):
            s = str(v)
            return s.rstrip("0").rstrip(".")
        return str(v)
    except Exception:
        return "0"

def pick_collection(row: dict) -> str:
    """
    Your SF REST currently returns: collectionName
    (optional fallback: Collection__c)
    """
    v = row.get("collectionName") or row.get("Collection__c")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return ""

def pick_message(row: dict) -> str:
    v = row.get("message")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return ""

def is_discontinued(row: dict) -> bool:
    """
    Your SF REST currently returns: isDiscontinued
    (optional fallback: discontinued or Name contains 'discont')
    """
    v = row.get("isDiscontinued")
    if isinstance(v, bool):
        return v

    v2 = row.get("discontinued")
    if isinstance(v2, bool):
        return v2

    nm = str(row.get("Name") or "")
    return ("discont" in nm.lower())

async def sf_get_brief(code_any: str) -> dict:
    code_sf = norm_lookup(code_any)
    items = await sf_product_search(code_sf)

    if DEBUG_SF:
        print("SF RAW RESPONSE:", items)

    row = items[0] if items else {}
    stockcode = str(row.get("stockcode") or code_sf)

    discontinued = is_discontinued(row)
    msg = pick_message(row)

    # If API didn't send a message but it is discontinued, we set one
    if discontinued and not msg:
        msg = "Discontinued item"

    return {
        "stockcode": stockcode,
        "collection": pick_collection(row),
        "qty": pick_qty(row),
        "name": str(row.get("Name") or ""),
        "discontinued": discontinued,
        "message": msg,
    }

def format_text(info: dict) -> str:
    code = info["stockcode"]
    col = info.get("collection", "")
    qty = info.get("qty", "0")

    msg = f"✅ {code}\n"
    if col:
        msg += f"Collection: {col}\n"
    msg += f"Available Qty: {qty}"

    # If discontinued -> prepend warning
    if info.get("discontinued"):
        warning = info.get("message") or "Discontinued item"
        msg = f"⚠️ {warning}\n\n" + msg

    return msg

def format_caption(info: dict) -> str:
    code = info["stockcode"]
    col = info.get("collection", "")
    qty = info.get("qty", "0")

    base = f"{code} | Qty:{qty}"
    if col:
        base = f"{code} | {col[:40]} | Qty:{qty}"

    if info.get("discontinued"):
        base = "DISCONTINUED | " + base

    return base[:1024]

# =========================================================
# Core flows
# =========================================================
async def handle_img_flow(sender: str, raw_after_prefix: str):
    info = await sf_get_brief(raw_after_prefix)

    # 1) send text info (includes collection + discontinued warning)
    await wa_send_text(sender, format_text(info))

    # 2) send image if available in excel map
    image_url = get_image_url_for_code(raw_after_prefix)
    if not image_url:
        await wa_send_text(sender, f"⚠️ Image not found for {norm_lookup(raw_after_prefix)} (Excel mapping missing).")
        return

    cache_key = norm_lookup(raw_after_prefix)
    media_id = _stock_to_media_id.get(cache_key)
    if not media_id:
        media_id = await wa_upload_image_from_url(image_url)
        _stock_to_media_id[cache_key] = media_id

    await wa_send_image_id(sender, media_id, caption=format_caption(info))

async def send_similar_results(sender: str, results: List[dict], title: str):
    if not results:
        await wa_send_text(sender, "❌ No similar images found.")
        return

    await wa_send_text(sender, title)

    sent = 0
    for row in results:
        sc_raw = str(row.get("stockcode") or row.get("stock_code") or row.get("code") or "")
        if not sc_raw:
            continue

        url = (row.get("image_url") or row.get("url") or "").strip()
        if not url:
            url = get_image_url_for_code(sc_raw) or ""
        if not url:
            continue

        info = await sf_get_brief(sc_raw)

        cache_key = norm_lookup(sc_raw)
        media_id = _stock_to_media_id.get(cache_key)
        if not media_id:
            media_id = await wa_upload_image_from_url(url)
            _stock_to_media_id[cache_key] = media_id

        await wa_send_image_id(sender, media_id, caption=format_caption(info))
        sent += 1
        if sent >= SIM_TOP_K:
            break

async def handle_similar_by_code(sender: str, raw_after_prefix: str):
    if _img_index is None or _clip_model is None:
        await wa_send_text(sender, "⚠️ Similarity system not ready (index/model not loaded).")
        return

    image_url = get_image_url_for_code(raw_after_prefix)
    if not image_url:
        await wa_send_text(sender, f"❌ No image URL found for {norm_lookup(raw_after_prefix)}")
        return

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(image_url)
        r.raise_for_status()
        img_bytes = r.content

    vec = embed_image_bytes(img_bytes)
    results = faiss_search(vec, k=SIM_TOP_K)

    await send_similar_results(sender, results, title=f"✅ Similar images for {norm_lookup(raw_after_prefix)}:")

async def handle_similar_by_uploaded_image(sender: str, media_id: str):
    if _img_index is None or _clip_model is None:
        await wa_send_text(sender, "⚠️ Similarity system not ready (index/model not loaded).")
        return

    img_bytes = await wa_download_media_bytes(media_id)
    vec = embed_image_bytes(img_bytes)
    results = faiss_search(vec, k=SIM_TOP_K)
    await send_similar_results(sender, results, title="✅ Similar images:")

# =========================================================
# QA fallback
# =========================================================
async def answer_user_text(text: str) -> str:
    try:
        doc_ans = ask_docs(text)
        if doc_ans:
            reply = (doc_ans.get("answer") or "").strip()
            sources = doc_ans.get("sources") or []
            if sources:
                reply += "\n\nSources:\n" + "\n".join(sources[:8])
            return reply or SAFE_FALLBACK_MESSAGE
    except Exception:
        safe_log_exc("[QA] ask_docs failed")

    try:
        return (ask_openai(text) or SAFE_FALLBACK_MESSAGE).strip()
    except Exception:
        safe_log_exc("[QA] ask_openai failed")
        return SAFE_FALLBACK_MESSAGE

# =========================================================
# Lifespan
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[BOOT] starting...")
    load_excel_map()
    load_image_similarity_index()
    load_clip_model()
    yield
    print("[BOOT] shutdown...")

app = FastAPI(title="NGC WhatsApp Bot", lifespan=lifespan)

# =========================================================
# Routes
# =========================================================
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/webhook/whatsapp")
async def whatsapp_verify(request: Request):
    qp = dict(request.query_params)
    mode = qp.get("hub.mode")
    token = qp.get("hub.verify_token")
    challenge = qp.get("hub.challenge")

    if mode == "subscribe" and token == META_VERIFY_TOKEN and challenge:
        return PlainTextResponse(str(challenge), status_code=200)

    return PlainTextResponse("Forbidden", status_code=403)

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    body = await request.body()
    verify_signature(body, request.headers.get("x-hub-signature-256"))

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)

    messages = extract_messages(payload)
    if not messages:
        return {"ok": True}

    for msg in messages:
        msg_id = get_msg_id(msg)
        if msg_id and msg_id in _seen:
            continue
        if msg_id:
            _seen[msg_id] = 1

        sender = get_sender(msg)
        if not sender:
            continue

        # A) Uploaded image -> similarity
        media_id = extract_image_media_id(msg)
        if media_id:
            try:
                await handle_similar_by_uploaded_image(sender, media_id)
            except Exception:
                safe_log_exc("[SIM] uploaded image flow failed")
                await wa_send_text(sender, SAFE_FALLBACK_MESSAGE)
            continue

        # B) Text
        text = extract_text(msg)
        if not text:
            continue

        # Parse commands first
        sim_raw = extract_sim_raw(text)
        if sim_raw is not None:
            try:
                await handle_similar_by_code(sender, sim_raw)
            except Exception:
                safe_log_exc("[SIM] sim-by-code failed")
                await wa_send_text(sender, SAFE_FALLBACK_MESSAGE)
            continue

        img_raw = extract_img_raw(text)
        if img_raw is not None:
            try:
                await handle_img_flow(sender, img_raw)
            except Exception:
                safe_log_exc("[IMG] img flow failed")
                await wa_send_text(sender, SAFE_FALLBACK_MESSAGE)
            continue

        # Normal QA/OpenAI
        try:
            reply = await answer_user_text(text)
            await wa_send_text(sender, reply)
        except Exception:
            safe_log_exc("[WEBHOOK] QA processing failed")
            await wa_send_text(sender, SAFE_FALLBACK_MESSAGE)

    return {"ok": True}