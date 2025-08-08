
"""qdrant_maintainer.py — merged maintainer
- Enrichment copied from qdbtest (robust scraping + company sites)
- Chunked embedding/upsert to Qdrant (like qdbtest)
- Per-collection delta logic (tester2 -> KG + Qdrant; veerive_docs -> Qdrant; prompt-guidance -> Qdrant)
"""

from __future__ import annotations

import os, re, time, hashlib, logging, traceback, urllib.parse
from typing import Dict, Any, List, Set, Optional
import schedule
from bson import ObjectId
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("db_sync.log"), logging.StreamHandler()])
logger = logging.getLogger("qdrant_maintainer")

load_dotenv()

MONGO_USERNAME = os.getenv("MONGO_USERNAME", "chaubeyp")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "ConsTrack360")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "veerive.tta8g.mongodb.net")
MONGO_DB = os.getenv("MONGO_DB", "veerive-db")

QDRANT_URL = os.getenv("QDRANT_URL", "https://9c4151fc-4aaf-418b-ac17-970854ac8a8f.europe-west3-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API", "")
REGULAR_COLL = os.getenv("QDRANT_COLLECTION", "tester2")
PDF_COLL = os.getenv("PDF_COLLECTION", "veerive_docs")
PROMPT_COLL = "prompt-guidance"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
SYNC_INTERVAL_MINUTES = int(os.getenv("SYNC_INTERVAL_MINUTES", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_SLEEP = float(os.getenv("RETRY_BASE_SLEEP", "2.0"))

def get_existing_ids_for_collection(client: QdrantClient, collection: str) -> set:
    ids = set()
    offset = None
    while True:
        points, offset = client.scroll(collection_name=collection, limit=100, offset=offset,
                                       with_payload=True, with_vectors=False)
        if not points:
            break
        for p in points:
            pl = p.payload or {}
            pid = None
            for k in ("postId","post_id","id","doc_id"):
                if k in pl and pl[k] is not None:
                    pid = pl[k]; break
            if pid is None:
                pid = p.id
            ids.add(str(pid))
        if offset is None:
            break
    logger.info(f"[Qdrant:{collection}] existing ids: {len(ids)}")
    return ids

class DatabaseSynchronizer:
    def __init__(self):
        self.mongo_client: Optional[MongoClient] = None
        self.db = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedder: Optional[OpenAIEmbeddings] = None
        self.chrome_options = None
        self._setup_browser_options()

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
            pwd = urllib.parse.quote_plus(MONGO_PASSWORD)
            uri = f"mongodb+srv://{MONGO_USERNAME}:{pwd}@{MONGO_CLUSTER}/"
            self.mongo_client = MongoClient(uri)
            self.db = self.mongo_client[MONGO_DB]
            logger.info("Connected to MongoDB")
            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            logger.info("Connected to Qdrant")
            # ensure regular collection exists
            names = [c.name for c in self.qdrant_client.get_collections().collections]
            if REGULAR_COLL not in names:
                self.qdrant_client.create_collection(collection_name=REGULAR_COLL,
                                                     vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))
                logger.info(f"Created Qdrant collection: {REGULAR_COLL}")
            self.embedder = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            traceback.print_exc()
            return False

    def _resolve_reference(self, collection_name, object_ids):
        if not object_ids: return []
        if not isinstance(object_ids, list): object_ids = [object_ids]
        valid = [ObjectId(x) for x in object_ids if x and ObjectId.is_valid(x)]
        if not valid: return []
        docs = list(self.db[collection_name].find({"_id": {"$in": valid}}))
        for d in docs: d["_id"] = str(d["_id"])
        return docs

    def _scrape_article(self, url: str) -> Dict[str, Any]:
        if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            return {}
        driver = None
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.set_page_load_timeout(50)
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            title = driver.title
            text = ""
            content_selectors = ["article", "main", ".article-content", ".post-content", "#content", ".content", ".entry-content", ".article-body"]
            for selector in content_selectors:
                try:
                    is_css = selector.startswith('.') or selector.startswith('#') or '[' in selector
                    elements = driver.find_elements(By.CSS_SELECTOR if is_css else By.TAG_NAME, selector)
                    for element in elements:
                        if not element.is_displayed(): continue
                        for tag in ['script','style','nav','header','footer','iframe']:
                            for junk in element.find_elements(By.TAG_NAME, tag):
                                driver.execute_script("arguments[0].remove()", junk)
                        if element.text.strip():
                            text += element.text.strip() + "\n\n"
                    if text: break
                except Exception:
                    continue
            if not text:
                try:
                    body = driver.find_element(By.TAG_NAME, "body")
                    for tag in ['script','style','nav','header','footer','iframe']:
                        for junk in body.find_elements(By.TAG_NAME, tag):
                            driver.execute_script("arguments[0].remove()", junk)
                    text = body.text
                except Exception:
                    pass
            authors, published_date = [], None
            for sel in ["[rel='author']", ".author", ".byline", "[itemprop='author']", ".author-name"]:
                try:
                    for el in driver.find_elements(By.CSS_SELECTOR, sel):
                        val = el.text.strip()
                        if val and val not in authors: authors.append(val)
                except Exception:
                    continue
            for sel in ["time", "[itemprop='datePublished']", "[property='article:published_time']", ".date", ".published-date"]:
                try:
                    for el in driver.find_elements(By.CSS_SELECTOR, sel):
                        published_date = el.get_attribute("datetime") or el.get_attribute("content") or el.text.strip()
                        if published_date: break
                    if published_date: break
                except Exception:
                    continue
            return {"title": title, "text": text, "authors": authors, "published_date": published_date}
        except Exception:
            return {}
        finally:
            if driver:
                try: driver.quit()
                except Exception: pass

    def get_all_enriched_posts(self, ids: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        coll = self.db["posts"]
        query = {}
        if ids:
            valid = [ObjectId(x) for x in ids if x and ObjectId.is_valid(x)]
            if valid: query = {"_id": {"$in": valid}}
        posts = list(coll.find(query).limit(limit)) if limit else list(coll.find(query))
        logger.info(f"Retrieved {len(posts)} posts for enrichment")
        enriched = []
        for post in posts:
            try:
                post["_id"] = str(post["_id"])
                post["contexts"] = self._resolve_reference("contexts", post.get("contexts", []))
                post["countries"] = self._resolve_reference("countries", post.get("countries", []))
                post["primaryCompanies"] = self._resolve_reference("companies", post.get("primaryCompanies", []))
                post["secondaryCompanies"] = self._resolve_reference("companies", post.get("secondaryCompanies", []))
                post["sectors"] = self._resolve_reference("sectors", post.get("sectors", []))
                post["subsectors"] = self._resolve_reference("subsectors", post.get("subsectors", []))

                source_refs = []
                if post.get("source") and ObjectId.is_valid(post["source"]):
                    source_refs = self._resolve_reference("sources", [post["source"]])
                post["source"] = source_refs[0] if source_refs else None

                post["scrapedArticle"] = self._scrape_article(post.get("sourceUrl", "")) if post.get("sourceUrl") else {}

                post["primarycompanydata"] = {}
                try:
                    if (post.get("primaryCompanies") and isinstance(post["primaryCompanies"], list) and
                        len(post["primaryCompanies"]) > 0 and isinstance(post["primaryCompanies"][0], dict)):
                        website = post["primaryCompanies"][0].get("website")
                        if website: post["primarycompanydata"] = self._scrape_article(website)
                except Exception: post["primarycompanydata"] = {}

                post["secondarycompanydata"] = {}
                try:
                    if (post.get("secondaryCompanies") and isinstance(post["secondaryCompanies"], list) and
                        len(post["secondaryCompanies"]) > 0 and isinstance(post["secondaryCompanies"][0], dict)):
                        website2 = post["secondaryCompanies"][0].get("website")
                        if website2: post["secondarycompanydata"] = self._scrape_article(website2)
                except Exception: post["secondarycompanydata"] = {}

                enriched.append(post)
            except Exception as e:
                logger.error(f"[Enrich] Failed {post.get('_id')}: {e}")
                continue
        return enriched

    def _clean_text(self, txt: str) -> str:
        if not txt: return ""
        txt = re.sub(r"<[^>]+>", "", txt)
        return re.sub(r"\s+", " ", txt).strip()

    def update_qdrant_chunks(self, posts: List[Dict[str, Any]]) -> bool:
        if not posts:
            return True
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            batch_points = []
            for post in posts:
                title = post.get("postTitle", "")
                summary = post.get("summary", "")
                contexts = " ".join([c.get("name", "") for c in post.get("contexts", []) if isinstance(c, dict)])
                article_text = post.get("scrapedArticle", {}).get("text", "")
                clean = self._clean_text(" ".join([title, contexts, summary, article_text]))
                if not clean: continue
                chunks = splitter.create_documents([clean])
                texts = [c.page_content for c in chunks]
                vectors = self.embedder.embed_documents(texts)
                for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                    payload = {
                        "postId": str(post.get("_id", "")),
                        "postTitle": title,
                        "chunk": chunk.page_content,
                        "chunkIndex": i,
                        "sourceUrl": post.get("sourceUrl", ""),
                        "summary": summary,
                        "primarydata": post.get("primarycompanydata", {}),
                        "secondarydata": post.get("secondarycompanydata", {}),
                        "contexts": [c.get("name", "") for c in post.get("contexts", []) if isinstance(c, dict)]
                    }
                    pid = int(hashlib.md5(f"{payload['postId']}::{i}".encode()).hexdigest(), 16) % (2**63)
                    batch_points.append(PointStruct(id=pid, vector=vec, payload=payload))
                if len(batch_points) >= 200:
                    ok = self._adaptive_upsert(batch_points)
                    batch_points = []
                    if not ok: logger.error("Failed to upsert a chunk batch.")
            if batch_points:
                self._adaptive_upsert(batch_points)
            return True
        except Exception as e:
            logger.error(f"Chunked upsert error: {e}")
            return False

    def _adaptive_upsert(self, points: List[PointStruct], attempt: int = 1) -> bool:
        if not points:
            return True
        try:
            self.qdrant_client.upsert(collection_name=REGULAR_COLL, points=points, wait=False)
            return True
        except Exception as e:
            if attempt >= 5:
                logger.error(f"Upsert failed (final) size={len(points)}: {e}")
                return False
            if len(points) > 1:
                mid = len(points)//2
                left = self._adaptive_upsert(points[:mid], attempt+1)
                right = self._adaptive_upsert(points[mid:], attempt+1)
                return left and right
            time.sleep(2.0 * attempt)
            return self._adaptive_upsert(points, attempt+1)

def run_maintenance() -> bool:
    sync = DatabaseSynchronizer()
    if not sync.connect(): return False

    # Regular posts: compute missing ids then enrich, upsert to Qdrant, push to KG
    existing_regular = get_existing_ids_for_collection(sync.qdrant_client, REGULAR_COLL)
    mongo_ids = [str(p["_id"]) for p in sync.db["posts"].find({}, {"_id": 1})]
    missing_regular = [pid for pid in mongo_ids if pid not in existing_regular]
    if missing_regular:
        enriched = sync.get_all_enriched_posts(ids=missing_regular)
        if enriched:
            sync.update_qdrant_chunks(enriched)
            try:
                import KnowledgeGraph.kgraph as kgraph
                kgraph.ingest_documents(enriched)
            except Exception as e:
                logger.error(f"KG ingest failed: {e}")
    else:
        logger.info("No new regular posts to upsert or push to KG.")

    # PDFs - only missing into Qdrant via pdfss
    existing_pdfs = get_existing_ids_for_collection(sync.qdrant_client, PDF_COLL)
    pdf_mongo_ids = [str(p["_id"]) for p in sync.db["posts"].find(
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

    # Prompts - only missing into Qdrant via prompt_guidance
    existing_prompts = get_existing_ids_for_collection(sync.qdrant_client, PROMPT_COLL)
    try:
        from prompt_guidance import PromptGuidanceHandler
        pgh = PromptGuidanceHandler()
        if hasattr(pgh, "upsert_missing"):
            pgh.upsert_missing(existing_prompts)
        else:
            pgh.run()
    except Exception as e:
        logger.error(f"Prompt guidance upsert failed: {e}")

    logger.info("Maintenance run complete.")
    return True


def run_once_with_retries() -> bool:
    """Run maintenance with retries/backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        ok = run_maintenance()
        if ok:
            logger.info("Maintenance completed successfully")
            return True
        sleep_s = RETRY_BASE_SLEEP * attempt
        logger.warning(f"Maintenance failed (attempt {attempt}/{MAX_RETRIES}). Retrying in {sleep_s:.1f}s...")
        time.sleep(sleep_s)
    logger.error("Maintenance failed after maximum retries.")
    return False

def run_scheduler():
    """Run recurring maintenance using 'schedule' with the configured interval."""
    logger.info(f"Starting scheduler: every {SYNC_INTERVAL_MINUTES} minutes")
    schedule.every(SYNC_INTERVAL_MINUTES).minutes.do(run_once_with_retries)
    # Kick off the first run immediately
    run_once_with_retries()
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")

if __name__ == "__main__":
    run_scheduler()