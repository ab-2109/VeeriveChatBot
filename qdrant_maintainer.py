"""qdrant_maintainer.py

Lean maintenance orchestrator for Veerive RAG pipeline.

Responsibilities:
 1. Upsert regular Mongo posts (tester2) into Qdrant (deduplicated) and push ONLY those to the knowledge graph via KnowledgeGraph.kgraph.
 2. Upsert PDF/table docs (veerive_docs) via pdfss.run_ingestion_filtered (no KG push).
 3. Upsert prompt guidance docs (prompt-guidance) via PromptGuidanceHandler (no KG push).

Removed legacy code:
 - Full sync scheduler / retry loops
 - Embedded Neo4j/LLM entity extraction (delegated to kgraph module)
 - Internal prompt guidance and PDF ingestion logic (delegated to prompt_guidance.py / pdfss.py)
 - Redundant KG construction code

Keep only: Mongo/Qdrant connection, enrichment (reference resolution + optional scraping), and batch embedding/upsert for regular posts.
"""
from __future__ import annotations

import os
import re
import time
import hashlib
import logging
import traceback
import urllib.parse
from typing import Dict, Any, List, Set, Optional
from datetime import datetime

from bson import ObjectId
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

# Optional scraping imports (used only if sourceUrl / company websites provided)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from langchain_openai import OpenAIEmbeddings

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("qdrant_maintainer")

# -----------------------------------------------------------------------------
# Environment / Config
# -----------------------------------------------------------------------------
load_dotenv()

MONGO_USERNAME = os.getenv("MONGO_USERNAME", "chaubeyp")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "ConsTrack360")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "veerive.tta8g.mongodb.net")
MONGO_DB = os.getenv("MONGO_DB", "veerive-db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "posts")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "tester2")  # regular posts
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
QDRANT_MAX_ATTEMPTS = int(os.getenv("QDRANT_MAX_ATTEMPTS", "5"))
QDRANT_RETRY_BASE_SLEEP = float(os.getenv("QDRANT_RETRY_BASE_SLEEP", "2.0"))

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def get_existing_ids_for_collection(client: QdrantClient, collection: str) -> set:
    """Scroll a collection and return a set of string IDs from payload (postId/post_id/id/doc_id) or point id."""
    ids = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        if not points:
            break
        for p in points:
            pl = p.payload or {}
            pid = None
            for k in ("postId", "post_id", "id", "doc_id"):
                if k in pl and pl[k] is not None:
                    pid = pl[k]; break
            if pid is None:
                pid = p.id
            ids.add(str(pid))
        if offset is None:
            break
    logger.info(f"[Qdrant:{collection}] existing ids: {len(ids)}")
    return ids

# -----------------------------------------------------------------------------
# Core minimal synchronizer (for regular documents only)
# -----------------------------------------------------------------------------
class DatabaseSynchronizer:
    def __init__(self):
        self.mongo_client: Optional[MongoClient] = None
        self.db = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedder: Optional[OpenAIEmbeddings] = None
        self.chrome_options = None
        self._setup_browser_options()

    # ---------------------- Connection & Setup -------------------------------
    def _setup_browser_options(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless=new")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0")
        prefs = {"profile.managed_default_content_settings.images": 2}
        self.chrome_options.add_experimental_option("prefs", prefs)

    def connect(self) -> bool:
        try:
            password = urllib.parse.quote_plus(MONGO_PASSWORD)
            uri = f"mongodb+srv://{MONGO_USERNAME}:{password}@{MONGO_CLUSTER}/"
            self.mongo_client = MongoClient(uri)
            self.db = self.mongo_client[MONGO_DB]
            logger.info("Connected to MongoDB")

            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            logger.info("Connected to Qdrant")

            # Ensure regular collection exists
            collections = self.qdrant_client.get_collections().collections
            if QDRANT_COLLECTION not in [c.name for c in collections]:
                self.qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {QDRANT_COLLECTION}")

            self.embedder = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
            logger.info(f"Initialized embedder: {EMBED_MODEL}")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            traceback.print_exc()
            return False

    # ---------------------- Enrichment / Helpers ----------------------------
    def _resolve_reference(self, collection_name, object_ids):
        if not object_ids:
            return []
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
        valid = [ObjectId(x) for x in object_ids if x and ObjectId.is_valid(x)]
        if not valid:
            return []
        docs = list(self.db[collection_name].find({"_id": {"$in": valid}}))
        for d in docs: d["_id"] = str(d["_id"])
        return docs

    def _scrape_article(self, url: str) -> Dict[str, Any]:
        if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            return {}
        driver = None
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.set_page_load_timeout(25)
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            title = driver.title
            body_text = driver.find_element(By.TAG_NAME, "body").text
            return {"title": title, "text": body_text}
        except Exception:
            return {}
        finally:
            if driver:
                try: driver.quit()
                except Exception: pass

    def enrich_post(self, post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            post["_id"] = str(post["_id"])
            post["contexts"] = self._resolve_reference("contexts", post.get("contexts", []))
            post["countries"] = self._resolve_reference("countries", post.get("countries", []))
            post["signals"] = self._resolve_reference("signals", post.get("signals", []))
            post["subsignals"] = self._resolve_reference("subsignals", post.get("subsignals", []))
            post["primaryCompanies"] = self._resolve_reference("companies", post.get("primaryCompanies", []))
            post["secondaryCompanies"] = self._resolve_reference("companies", post.get("secondaryCompanies", []))
            post["sectors"] = self._resolve_reference("sectors", post.get("sectors", []))
            post["subsectors"] = self._resolve_reference("subsectors", post.get("subsectors", []))
            post["themes"] = self._resolve_reference("themes", post.get("themes", []))
            post["regions"] = self._resolve_reference("regions", post.get("regions", []))

            # Source
            if post.get("source") and ObjectId.is_valid(post.get("source")):
                src = self._resolve_reference("sources", [post["source"]])
                post["source"] = src[0] if src else None
            else:
                post["source"] = None

            # Optional scraping
            if post.get("sourceUrl"):
                post["scrapedArticle"] = self._scrape_article(post["sourceUrl"])
            else:
                post["scrapedArticle"] = {}
            return post
        except Exception as e:
            logger.error(f"Enrich error {post.get('_id')}: {e}")
            return None

    # ---------------------- Vectorization / Upsert --------------------------
    @staticmethod
    def _clean_text(txt: str) -> str:
        if not txt: return ""
        txt = re.sub(r"<[^>]+>", "", txt)
        return re.sub(r"\s+", " ", txt).strip()

    def _prepare_vector_doc(self, post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = post.get("postTitle", "")
        summary = post.get("summary", "")
        content = post.get("content", "")
        contexts = " ".join([c.get("name", "") for c in post.get("contexts", []) if isinstance(c, dict)])
        article_text = post.get("scrapedArticle", {}).get("text", "")
        combined = self._clean_text(f"{title} {contexts} {summary} {content} {article_text}")
        if not combined:
            return None
        return {
            "post_id": post["_id"],
            "title": title,
            "text_for_embedding": combined,
            "summary": summary,
            "source_url": post.get("sourceUrl", ""),
            "contexts": [c.get("name", "") for c in post.get("contexts", []) if isinstance(c, dict)],
            "countries": [c.get("name", "") for c in post.get("countries", []) if isinstance(c, dict)],
            "sectors": [c.get("name", "") for c in post.get("sectors", []) if isinstance(c, dict)],
            "subsectors": [c.get("name", "") for c in post.get("subsectors", []) if isinstance(c, dict)],
            "primary_companies": [c.get("name", "") for c in post.get("primaryCompanies", []) if isinstance(c, dict)],
            "secondary_companies": [c.get("name", "") for c in post.get("secondaryCompanies", []) if isinstance(c, dict)],
            "created_at": str(post.get("createdAt", "")),
            "updated_at": str(post.get("updatedAt", ""))
        }

    def update_qdrant(self, documents: List[Dict[str, Any]]) -> bool:
        if not documents:
            return True
        try:
            batches = [documents[i:i+BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]
            for i, batch in enumerate(batches):
                logger.info(f"Embedding batch {i+1}/{len(batches)} size={len(batch)}")
                vec_ready = [self._prepare_vector_doc(d) for d in batch]
                vec_ready = [v for v in vec_ready if v]
                if not vec_ready:
                    continue
                texts = [v["text_for_embedding"] for v in vec_ready]
                vectors = self.embedder.embed_documents(texts)
                points = []
                for v, emb in zip(vec_ready, vectors):
                    pid = int(hashlib.md5(v["post_id"].encode()).hexdigest(), 16) % (2**63)
                    payload = {k: val for k, val in v.items() if k != "text_for_embedding"}
                    points.append(PointStruct(id=pid, vector=emb, payload=payload))
                # Adaptive upsert with retry & micro-batching
                if not self._adaptive_upsert(points):
                    logger.error(f"Failed to upsert batch {i+1}/{len(batches)} after retries")
                else:
                    logger.info(f"Upserted {len(points)} points to {QDRANT_COLLECTION}")
            return True
        except Exception as e:
            logger.error(f"Qdrant upsert error: {e}")
            traceback.print_exc()
            return False

    def _adaptive_upsert(self, points: List[PointStruct], attempt: int = 1) -> bool:
        """Attempt to upsert a list of points with retries and recursive splitting on timeout.
        Returns True if all points eventually succeed, False otherwise.
        """
        if not points:
            return True
        try:
            self.qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points,
                wait=False  # avoid waiting for index persistence to reduce timeout risk
            )
            return True
        except Exception as e:
            if attempt >= QDRANT_MAX_ATTEMPTS:
                logger.error(f"Upsert failed (final) size={len(points)}: {e}")
                return False
            # If more than one point, split and retry halves to isolate problematic size/timeouts
            if len(points) > 1:
                mid = len(points) // 2
                logger.warning(f"Upsert timeout/err (attempt {attempt}) size={len(points)} -> splitting into {mid} + {len(points)-mid}")
                left = self._adaptive_upsert(points[:mid], attempt=attempt+1)
                right = self._adaptive_upsert(points[mid:], attempt=attempt+1)
                return left and right
            # Single point retry with backoff
            sleep_s = QDRANT_RETRY_BASE_SLEEP * attempt
            logger.warning(f"Retrying single point (attempt {attempt}) after {sleep_s:.1f}s: {e}")
            time.sleep(sleep_s)
            return self._adaptive_upsert(points, attempt=attempt+1)

# -----------------------------------------------------------------------------
# Maintenance entry point
# -----------------------------------------------------------------------------

def run_maintenance() -> bool:
    """Main entry: sync Mongo -> Qdrant for 3 collections; push tester2 posts to KG only."""
    sync = DatabaseSynchronizer()
    if not sync.connect():
        return False

    coll_regular = "tester2"
    coll_pdfs = "veerive_docs"
    coll_prompts = "prompt-guidance"

    # 1) Regular posts (tester2): find missing IDs and upsert to qdrant + push to KG
    # existing_regular = get_existing_ids_for_collection(sync.qdrant_client, coll_regular)
    # mongo_ids = [str(p["_id"]) for p in sync.db[MONGO_COLLECTION].find({}, {"_id": 1})]
    # missing_regular = [pid for pid in mongo_ids if pid not in existing_regular]

    # if missing_regular:
    #     posts = sync.db[MONGO_COLLECTION].find({"_id": {"$in": [ObjectId(x) for x in missing_regular]}})
    #     enriched: List[Dict[str, Any]] = []
    #     for post in posts:
    #         ep = sync.enrich_post(post)
    #         if ep:
    #             enriched.append(ep)
    #     sync.update_qdrant(enriched)
    #     # Knowledge graph ingest (external module)
    #     try:
    #         import KnowledgeGraph.kgraph as kgraph
    #         kgraph.ingest_documents(enriched)
    #     except Exception as e:
    #         logger.error(f"KG ingest failed: {e}")
    # else:
    #     logger.info("No new regular posts to upsert or push to KG.")

    # 2) PDFs/Tables (veerive_docs): only upsert to Qdrant via pdfss helper
    existing_pdfs = get_existing_ids_for_collection(sync.qdrant_client, coll_pdfs)
    pdf_mongo_ids = [str(p["_id"]) for p in sync.db[MONGO_COLLECTION].find(
        {"googleDriveUrl": {"$exists": True, "$ne": None, "$ne": ""}}, {"_id": 1}
    )]
    missing_pdfs = set(pdf_mongo_ids) - existing_pdfs
    if missing_pdfs:
        try:
            import pdfss
            processed = pdfss.run_ingestion_filtered(set(ObjectId(x) for x in missing_pdfs))
            logger.info(f"PDF ingestion complete. Processed: {processed}")
        except Exception as e:
            logger.error(f"PDF ingestion failed: {e}")
    else:
        logger.info("No new PDF/table docs to upsert.")

    # 3) Prompt guidance (prompt-guidance): only upsert to Qdrant
    existing_prompts = get_existing_ids_for_collection(sync.qdrant_client, coll_prompts)
    try:
        from prompt_guidance import PromptGuidanceHandler
        pgh = PromptGuidanceHandler()
        pgh.upsert_missing(existing_prompts)  # Requires method existence; fallback to run() if absent
    except AttributeError:
        try:
            pgh = PromptGuidanceHandler()
            pgh.run()
        except Exception as e:
            logger.error(f"Prompt guidance processing failed: {e}")
    except Exception as e:
        logger.error(f"Prompt guidance upsert failed: {e}")

    logger.info("Maintenance run complete.")
    return True

if __name__ == "__main__":
    run_maintenance()