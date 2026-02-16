# app.py
import os
import json
import hmac
import hashlib
import traceback
import re
from typing import Optional, Dict, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from qa import ask_docs, ask_openai
from salesforce_products import sf_product_search  # auth/token handled outside

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
DEBUG_SF = os.getenv("DEBUG_SF", "false").lower() == "true"

SAFE_FALLBACK_MESSAGE = os.getenv(
    "SAFE_FALLBACK_MESSAGE",
    "Sorry, I’m having trouble right now. Please try again later.",
).strip()

# Set true while testing if signature verification blocks you
SKIP_SIGNATURE_VERIFY = os.getenv("SKIP_SIGNATURE_VERIFY", "false").lower() == "true"

# Dedupe to avoid double replies
_seen: Dict[str, int] = {}

# Extract ONLY the part AFTER IMG-
# e.g., "IMG-VF01" -> "VF01"
IMG_PATTERN = re.compile(r"\bIMG-([A-Za-z0-9_-]+)\b", re.IGNORECASE)


def verify_signature(body: bytes, header: Optional[str]):
    """
    Meta sends: x-hub-signature-256: sha256=<hmac>
    """
    if SKIP_SIGNATURE_VERIFY:
        return
    if not META_APP_SECRET:
        # production should set this; allow in dev
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


def extract_img_code(text: str) -> Optional[str]:
    """
    Returns the string AFTER IMG- (e.g., IMG-VF01 -> VF01).
    """
    m = IMG_PATTERN.search(text or "")
    return m.group(1) if m else None


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
            print("Meta send error:", resp.status_code, resp.text)


def format_single_product_reply(display_code: str, items: list) -> str:
    """
    Show ONLY the asked code and quantity.
    """
    if not items:
        return f"❌ No active product found for: {display_code}"

    it = items[0]  # expect ONE item from Salesforce
    qty = it.get("quantity", 0)
    so_qty = it.get("so_qty", 0)

    return f"✅ {display_code}\nAvailable Qty: {qty}\nSO Qty: {so_qty}"


async def answer_user_question(q: str) -> str:
    """
    Priority:
    1) If message contains IMG-<code> -> call Salesforce inventory search using ONLY <code>
    2) Else -> SOP docs QA
    3) Else -> OpenAI fallback
    """
    try:
        img_code = extract_img_code(q)
        if img_code:
            # Send ONLY the part after IMG- to Salesforce (e.g., VF01)
            items = await sf_product_search(img_code)
            if DEBUG_SF:
                print("SF RAW RESPONSE:", items)

            # Display full code back to user
            display_code = f"IMG-{img_code}"
            return format_single_product_reply(display_code, items)

        # SOP-based QA
        doc_ans = ask_docs(q)
        if doc_ans:
            reply = (doc_ans.get("answer") or "").strip()
            sources = doc_ans.get("sources") or []
            if sources:
                reply += "\n\nSources:\n" + "\n".join(sources[:8])
            return reply or SAFE_FALLBACK_MESSAGE

        # General fallback
        return (ask_openai(q) or SAFE_FALLBACK_MESSAGE).strip()

    except Exception:
        traceback.print_exc()
        return SAFE_FALLBACK_MESSAGE


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

    # Verify signature (optional in dev)
    verify_signature(body, request.headers.get("x-hub-signature-256"))

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)

    print("INCOMING POST payload:", payload)

    messages = extract_messages(payload)
    if not messages:
        # statuses, delivery receipts, etc.
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

        reply = await answer_user_question(text)
        try:
            await wa_send_text(sender, reply)
        except Exception:
            traceback.print_exc()

    return {"ok": True}
