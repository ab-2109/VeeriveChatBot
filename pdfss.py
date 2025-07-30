import os
import io
import json
import pdfplumber
import pandas as pd
import tempfile
import urllib.parse
from dotenv import load_dotenv

from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
# === LOAD CONFIG ===
load_dotenv()

username = "chaubeyp"
password = urllib.parse.quote_plus("ConsTrack360")
MONGO_URI = f"mongodb+srv://{username}:{password}@veerive.tta8g.mongodb.net/"
DB_NAME = "veerive-db"
COLLECTION_NAME = "posts"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API")
QDRANT_COLLECTION = "veerive_docs"

SERVICE_ACCOUNT_FILE = "veerivechatbot-c34f38f5e526.json"
client = OpenAI()

# === INIT CLIENTS ===
mongo = MongoClient(MONGO_URI)
collection = mongo[DB_NAME][COLLECTION_NAME]

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

try:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    print(f"[INFO] Created Qdrant collection: {QDRANT_COLLECTION}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"[INFO] Qdrant collection '{QDRANT_COLLECTION}' already exists")
    else:
        print(f"[ERROR] Could not create collection: {e}")

# === INIT GOOGLE DRIVE ===
with open(SERVICE_ACCOUNT_FILE, 'r') as f:
    service_account_info = json.load(f)
    service_account_email = service_account_info['client_email']
print(f"Service Account Email: {service_account_email}")

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
drive_service = build('drive', 'v3', credentials=credentials)

# === HELPERS ===

def extract_file_id(gdrive_url):
    try:
        return gdrive_url.split("/d/")[1].split("/")[0]
    except Exception:
        return None

def download_pdf(file_id):
    file_info = drive_service.files().get(fileId=file_id).execute()
    print(f"[INFO] Downloading: {file_info.get('name', 'Unknown')}")

    request = drive_service.files().get_media(fileId=file_id)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    with io.FileIO(temp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return temp_path

def extract_text_and_tables(pdf_path):
    full_text = ""
    table_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"[INFO] Processing page {i + 1}")
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            tables = page.extract_tables()
            for table in tables:
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    table_data.append(df.to_markdown(index=False))

    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150)
    text_chunks = splitter.split_text(full_text)

    return text_chunks, table_data

def embed_chunks(chunks):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=chunks
    )
    return [item.embedding for item in response.data]

import uuid
import time
from qdrant_client.http.exceptions import UnexpectedResponse

def upsert_to_qdrant(chunks, embeddings, meta_base, is_table=False, batch_size=15):
    total = len(chunks)
    for batch_start in range(0, total, batch_size):
        batch_points = []
        for i in range(batch_start, min(batch_start + batch_size, total)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    "chunk_text": chunks[i],
                    "chunk_id": i,
                    "is_table": is_table,
                    **meta_base
                }
            )
            batch_points.append(point)

        # Retry logic
        success = False
        attempts = 0
        while not success and attempts < 3:
            try:
                print(f"[INFO] Uploading batch {batch_start}–{batch_start + len(batch_points)}...")
                qdrant.upsert(collection_name=QDRANT_COLLECTION, points=batch_points)
                success = True
            except Exception as e:
                attempts += 1
                print(f"[WARNING] Qdrant upsert failed (attempt {attempts}): {e}")
                time.sleep(5)

        if not success:
            raise RuntimeError(f"[ERROR] Qdrant upsert failed permanently for batch {batch_start}")


# === MAIN INGESTION ===

def run_ingestion():
    total_docs = collection.count_documents({})
    print(f"[INFO] Total documents in collection: {total_docs}")

    docs = collection.find({
        "googleDriveUrl": {"$exists": True, "$ne": None, "$ne": ""}
    })

    processed = 0
    permission_denied = 0

    for doc in docs:
        url = doc.get("googleDriveUrl", "")
        title = doc.get("postTitle", "Untitled")
        file_id = extract_file_id(url)

        if not file_id:
            print(f"[!] Skipping invalid URL: {url}")
            continue

        try:
            print(f"[+] Processing: {title}")
            pdf_path = download_pdf(file_id)
            text_chunks, table_chunks = extract_text_and_tables(pdf_path)

            meta_base = {
                "doc_id": str(doc["_id"]),
                "doc_title": title,
                "source_url": url,
                "post_type": doc.get("postType", ""),
                "sentiment": doc.get("sentiment", ""),
                "source": "gdrive"
            }

            # Text chunks
            if text_chunks:
                text_embeddings = embed_chunks(text_chunks)
                upsert_to_qdrant(text_chunks, text_embeddings, meta_base, is_table=False)

            # Table chunks
            if table_chunks:
                table_embeddings = embed_chunks(table_chunks)
                upsert_to_qdrant(table_chunks, table_embeddings, meta_base, is_table=True)

            collection.update_one({"_id": doc["_id"]}, {"$set": {"embedded": True}})
            processed += 1

        except Exception as e:
            if "File not found" in str(e) or "404" in str(e):
                print(f"[!] Permission denied for {url}")
                permission_denied += 1
                collection.update_one({"_id": doc["_id"]}, {"$set": {"permission_error": True}})
            else:
                print(f"[X] Failed for {url} → {e}")

    print(f"[INFO] ✅ Processed {processed} documents.")
    if permission_denied:
        print(f"[WARNING] ❌ {permission_denied} documents had permission issues.")

# === ENTRY POINT ===

if __name__ == "__main__":
    run_ingestion()
