# app.py
import os
import json
import hmac
import hashlib
import traceback
import re
import mimetypes
from urllib.parse import urlparse
from typing import Optional, Dict, List

import httpx
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from qa import ask_docs, ask_openai
from salesforce_products import sf_product_search  # your existing async SF call

load_dotenv()
app = FastAPI(title="NGC WhatsApp Bot")

# =========================================================
# ENV – WhatsApp Cloud API
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
# Excel mapping (RAW stock_code -> image_url)
# Excel MUST store raw code WITHOUT "IMG-"
# columns by default: stock_code, image_url
# =========================================================
IMAGE_XLSX_PATH = os.getenv("IMAGE_XLSX_PATH", "data/Full_stock_Cleaned.xlsx").strip()
EXCEL_STOCK_COL = os.getenv("EXCEL_STOCK_COL", "stock_code").strip()
EXCEL_URL_COL = os.getenv("EXCEL_URL_COL", "image_url").strip()

# Cache in memory
_stock_to_url: Dict[str, str] = {}
# Optional cache to avoid re-uploading same image repeatedly
_stock_to_media_id: Dict[str, str] = {}

# Dedupe message IDs
_seen: Dict[str, int] = {}

# STRICT: triggers ONLY if whole message is like "IMG-xxxx" (spaces allowed)
IMG_PATTERN = re.compile(r"^\s*IMG-\s*([A-Za-z0-9_-]+)\s*$", re.IGNORECASE)


# =========================================================
# Helpers
# =========================================================
def _norm_code(s: str) -> str:
    return (s or "").strip().upper()

def extract_img_code(text: str) -> Optional[str]:
    """
    Returns raw code only.
    "IMG- SJC305-1042" -> "SJC305-1042"
    Only matches if the entire message is IMG-xxxx
    """
    m = IMG_PATTERN.match(text or "")
    print("EXTRACTED RAW CODE:", m.group(1).strip().upper() if m else None)
    return m.group(1).strip().upper() if m else None

def load_excel_map() -> int:
    """
    Loads Excel into memory:
      RAW_CODE -> image_url
    If Excel contains "IMG-" by mistake, we strip it during load.
    """
    global _stock_to_url

    if not os.path.exists(IMAGE_XLSX_PATH):
        print(f"[WARN] Excel not found at: {IMAGE_XLSX_PATH}")
        _stock_to_url = {}
        return 0

    df = pd.read_excel(IMAGE_XLSX_PATH)
    if EXCEL_STOCK_COL not in df.columns or EXCEL_URL_COL not in df.columns:
        raise RuntimeError(
            f"Excel must contain columns: {EXCEL_STOCK_COL}, {EXCEL_URL_COL}. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=[EXCEL_STOCK_COL, EXCEL_URL_COL]).copy()

    # Normalize codes: strip IMG- if present, uppercase, trim
    df[EXCEL_STOCK_COL] = (
        df[EXCEL_STOCK_COL]
        .astype(str)
        .str.replace(r"^\s*IMG-\s*", "", regex=True)
        .map(_norm_code)
    )
    df[EXCEL_URL_COL] = df[EXCEL_URL_COL].astype(str).str.strip()

    # remove duplicates (keep first)
    df = df.drop_duplicates(subset=[EXCEL_STOCK_COL], keep="first")

    _stock_to_url = dict(zip(df[EXCEL_STOCK_COL], df[EXCEL_URL_COL]))
    print(f"[OK] Loaded {len(_stock_to_url)} image mappings from Excel.")
    return len(_stock_to_url)

def get_image_url_for_raw_code(raw_code: str) -> Optional[str]:
    """
    Excel stores ONLY RAW code (no IMG-).
    """
    if not _stock_to_url:
        try:
            load_excel_map()
        except Exception as e:
            print("[ERROR] Excel load failed:", e)
            return None
    return _stock_to_url.get(_norm_code(raw_code))

def verify_signature(body: bytes, header: Optional[str]):
    """
    Meta sends: x-hub-signature-256: sha256=<hmac>
    """
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

def extract_messages(payload: dict) -> List[dict]:
    msgs: List[dict] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value") or {}
            for msg in (value.get("messages") or []):
                msgs.append(msg)
    return msgs

def extract_text(msg: dict) -> str:
    if msg.get("type") == "text":
        return (msg.get("text") or {}).get("body", "").strip()
    return ""

def get_sender(msg: dict) -> str:
    return (msg.get("from") or "").strip()

def get_msg_id(msg: dict) -> str:
    return (msg.get("id") or "").strip()


# =========================================================
# WhatsApp send helpers
# =========================================================
async def wa_send_text(to: str, text: str):
    if not (META_ACCESS_TOKEN and META_PHONE_NUMBER_ID):
        raise RuntimeError("META_ACCESS_TOKEN / META_PHONE_NUMBER_ID not set")

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text[:4096]},
    }

    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            print("Meta send text error:", resp.status_code, resp.text)

async def wa_upload_image_from_url(image_url: str) -> str:
    """
    Download image from URL -> upload to WhatsApp media endpoint -> return media_id
    (This sends image as an image, NOT a URL message.)
    """
    if not (META_ACCESS_TOKEN and META_PHONE_NUMBER_ID):
        raise RuntimeError("META_ACCESS_TOKEN / META_PHONE_NUMBER_ID not set")

    # 1) download
    async with httpx.AsyncClient(timeout=40) as client:
        img_resp = await client.get(image_url)
        img_resp.raise_for_status()
        img_bytes = img_resp.content

    # 2) guess mime
    path = urlparse(image_url).path
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"

    # 3) upload
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{META_PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
    files = {"file": ("image", img_bytes, mime)}
    data = {"messaging_product": "whatsapp"}

    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.post(url, headers=headers, data=data, files=files)
        resp.raise_for_status()
        return resp.json()["id"]

async def wa_send_image_id(to: str, media_id: str, caption: str = ""):
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
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

def format_single_product_reply(display_code: str, items: list) -> str:
    """
    Show only asking code & qty.
    Your sf_product_search returns list of dicts with fields like quantity/so_qty.
    """
    if not items:
        return f"❌ No active product found for: {display_code}"

    it = items[0]
    qty = it.get("quantity", 0)
    so_qty = it.get("so_qty", 0)
    return f"✅ {display_code}\nAvailable Qty: {qty}\nSO Qty: {so_qty}"


# =========================================================
# Core logic
# =========================================================
async def handle_img_flow(sender: str, raw_code: str):
    """
    raw_code = code after IMG- (no IMG- included)
    1) get qty from Salesforce (using raw_code)
    2) send text
    3) get image_url from Excel using raw_code
    4) upload image -> media_id
    5) send image
    """
    # Salesforce qty
    items = await sf_product_search(raw_code)
    if DEBUG_SF:
        print("SF RAW RESPONSE:", items)

    display_code = raw_code
    reply_text = format_single_product_reply(display_code, items)
    await wa_send_text(sender, reply_text)
    print(f"row_code={raw_code} qty={items[0].get('quantity', 0) if items else 'N/A'} sent_text_reply")

    # Excel mapping (RAW only)
    image_url = get_image_url_for_raw_code(raw_code)
    if not image_url:
        await wa_send_text(sender, f"⚠️ Image not found for {display_code} (Excel mapping missing).")
        return

    # cache media_id by raw code (not IMG-)
    cache_key = _norm_code(raw_code)
    media_id = _stock_to_media_id.get(cache_key)

    if not media_id:
        try:
            media_id = await wa_upload_image_from_url(image_url)
            _stock_to_media_id[cache_key] = media_id
        except Exception as e:
            print("Image upload failed:", e)
            await wa_send_text(sender, f"⚠️ Unable to send image for {display_code}.")
            return

    await wa_send_image_id(sender, media_id, caption=display_code)

async def answer_user_question(q: str) -> str:
    """
    Priority:
    - if strict IMG-xxxx -> returns internal marker
    - else -> docs QA
    - else -> OpenAI fallback
    """
    try:
        raw_code = extract_img_code(q)
        if raw_code:
            return "__IMG_FLOW__:" + raw_code  # internal marker

        doc_ans = ask_docs(q)
        if doc_ans:
            reply = (doc_ans.get("answer") or "").strip()
            sources = doc_ans.get("sources") or []
            if sources:
                reply += "\n\nSources:\n" + "\n".join(sources[:8])
            return reply or SAFE_FALLBACK_MESSAGE

        return (ask_openai(q) or SAFE_FALLBACK_MESSAGE).strip()

    except Exception:
        traceback.print_exc()
        return SAFE_FALLBACK_MESSAGE


# =========================================================
# Routes
# =========================================================
@app.on_event("startup")
async def on_startup():
    # Load Excel mapping once at startup (optional, but faster)
    try:
        load_excel_map()
    except Exception as e:
        print("[WARN] Excel map not loaded on startup:", e)

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
        text = extract_text(msg)

        if not sender or not text:
            continue

        try:
            reply = await answer_user_question(text)

            if reply.startswith("__IMG_FLOW__:"):
                raw_code = reply.split(":", 1)[1].strip()
                await handle_img_flow(sender, raw_code)
            else:
                await wa_send_text(sender, reply)

        except Exception:
            traceback.print_exc()
            try:
                await wa_send_text(sender, SAFE_FALLBACK_MESSAGE)
            except Exception:
                traceback.print_exc()

    return {"ok": True}
