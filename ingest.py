"""
ingest.py (Option 1 – Build vectorstore locally on EC2)

1) Reads SOP PDFs from:  s3://S3_BUCKET/sop/
2) Chunks text
3) Builds FAISS vectorstore using OpenAI embeddings
4) Saves locally to VECTORSTORE_DIR
"""

import os
import json
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import boto3
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# =========================================================
# ENV
# =========================================================
AWS_REGION = os.getenv("AWS_REGION", "me-central-1").strip()
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_SOP_PREFIX = os.getenv("S3_SOP_PREFIX", "sop/").strip()

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

CHUNK_SIZE = int(os.getenv("SOP_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("SOP_CHUNK_OVERLAP", "150"))

# IMPORTANT: Option 1 → NEVER upload vectorstore
UPLOAD_VECTORSTORE = False

s3 = boto3.client("s3", region_name=AWS_REGION)


def get_embeddings():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)


def list_pdf_keys(bucket: str, prefix: str) -> List[str]:
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            if obj["Key"].lower().endswith(".pdf"):
                keys.append(obj["Key"])
    return keys


def main():
    if not S3_BUCKET:
        raise SystemExit("Missing S3_BUCKET env var")

    keys = list_pdf_keys(S3_BUCKET, S3_SOP_PREFIX)
    if not keys:
        raise SystemExit(f"No SOP PDFs found in s3://{S3_BUCKET}/{S3_SOP_PREFIX}")

    print(f"Found {len(keys)} SOP PDFs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    embeddings = get_embeddings()
    all_docs = []
    manifest_docs: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)

        for key in keys:
            local_pdf = tmpdir / Path(key).name
            s3.download_file(S3_BUCKET, key, str(local_pdf))
            print("Downloaded:", key)

            loader = PyPDFLoader(str(local_pdf))
            docs = loader.load()

            filename = Path(key).name
            manifest_docs.append({
                "file": filename,
                "s3_key": key,
                "pages": len(docs),
            })

            for d in docs:
                d.metadata["source"] = filename
                d.metadata["s3_key"] = key

            all_docs.extend(docs)

    chunks = splitter.split_documents(all_docs)
    print("Total chunks:", len(chunks))

    vs = FAISS.from_documents(chunks, embeddings)

    outdir = Path(VECTORSTORE_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(outdir))

    manifest = {
        "created_at": int(time.time()),
        "embedding_model": OPENAI_EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "docs": manifest_docs,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Vectorstore saved to:", outdir.resolve())


if __name__ == "__main__":
    main()
