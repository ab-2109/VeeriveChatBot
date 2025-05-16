from pymongo import MongoClient
from bson.objectid import ObjectId
import urllib.parse
import pprint
import uuid
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

load_dotenv()

class QdrantConfig:
    MONGODB_USERNAME = "chaubeyp"
    MONGODB_PASSWORD = "ConsTrack360"
    MONGODB_URI = f"mongodb+srv://{MONGODB_USERNAME}:{urllib.parse.quote_plus(MONGODB_PASSWORD)}@veerive.tta8g.mongodb.net/"
    QDRANT_URL = "https://9c4151fc-4aaf-418b-ac17-970854ac8a8f.europe-west3-0.gcp.cloud.qdrant.io:6333"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RYAYHgpWnTy9SDZEkpER_1O_QSrvfZ-XTcrq8Wdhkx4"
    QDRANT_COLLECTION = "tester2"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS = 3072

class QdrantHandler:
    def __init__(self):
        self.mongo_client = MongoClient(QdrantConfig.MONGODB_URI)
        self.db = self.mongo_client["veerive-db"]
        self.qdrant_client = QdrantClient(url=QdrantConfig.QDRANT_URL, api_key=QdrantConfig.QDRANT_API_KEY)
        self.embedding = OpenAIEmbeddings(model=QdrantConfig.EMBEDDING_MODEL, openai_api_key=QdrantConfig.OPENAI_API_KEY)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self._ensure_collection_exists()

        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        self.chrome_options.add_experimental_option("prefs", {
            "profile.managed_default_content_settings.images": 2,
            "profile.managed_default_content_settings.javascript": 1,
            "profile.default_content_setting_values.notifications": 2,
            "profile.managed_default_content_settings.plugins": 2,
        })

    def _ensure_collection_exists(self):
        if QdrantConfig.QDRANT_COLLECTION not in [c.name for c in self.qdrant_client.get_collections().collections]:
            self.qdrant_client.recreate_collection(
                collection_name=QdrantConfig.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=QdrantConfig.EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE)
            )

    def clean_text(self, raw):
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", raw or "")).strip()

    def resolve_reference(self, collection_name, object_ids):
        if not object_ids: return []
        if not isinstance(object_ids, list): object_ids = [object_ids]
        valid_ids = [ObjectId(oid) for oid in object_ids if oid and ObjectId.is_valid(oid)]
        return list(self.db[collection_name].find({"_id": {"$in": valid_ids}})) if valid_ids else []
    
    def scrape_article(self, url):
        if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            print(f"[Invalid URL] {url}")
            return {}
        driver = None
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.set_page_load_timeout(50)
            print(f"[Scraping] {url}")
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            print(f"[Loaded] {url}")

            title = driver.title
            text = ""
            content_selectors = [
                "article", "main", ".article-content", ".post-content",
                "#content", ".content", ".entry-content", ".article-body"
            ]
            for selector in content_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR if selector.startswith('.') or selector.startswith('#') else By.TAG_NAME, selector)
                    for element in elements:
                        if not element.is_displayed(): continue
                        for tag in ['script', 'style', 'nav', 'header', 'footer', 'iframe']:
                            for junk in element.find_elements(By.TAG_NAME, tag):
                                driver.execute_script("arguments[0].remove()", junk)
                        if element.text.strip(): text += element.text.strip() + "\n\n"
                    if text: break
                except: continue

            if not text:
                try:
                    body = driver.find_element(By.TAG_NAME, "body")
                    for tag in ['script', 'style', 'nav', 'header', 'footer', 'iframe']:
                        for junk in body.find_elements(By.TAG_NAME, tag):
                            driver.execute_script("arguments[0].remove()", junk)
                    text = body.text
                except: pass

            authors, published_date = [], None
            for sel in ["[rel='author']", ".author", ".byline", "[itemprop='author']", ".author-name"]:
                try:
                    for el in driver.find_elements(By.CSS_SELECTOR, sel):
                        val = el.text.strip()
                        if val and val not in authors: authors.append(val)
                except: continue

            for sel in ["time", "[itemprop='datePublished']", "[property='article:published_time']", ".date", ".published-date"]:
                try:
                    for el in driver.find_elements(By.CSS_SELECTOR, sel):
                        published_date = el.get_attribute("datetime") or el.get_attribute("content") or el.text.strip()
                        if published_date: break
                    if published_date: break
                except: continue

            return {
                "title": title,
                "text": text,
                "authors": authors,
                "published_date": published_date
            }
        except Exception as e:
            print(f"[Error scraping] {url}: {str(e)}")
            return {}
        finally:
            if driver:
                try: driver.quit()
                except: pass

    def get_all_enriched_posts(self, limit=None):
        """Fetch posts and resolve all references to create enriched documents"""
        # Convert cursor to list immediately to avoid timeout
        cursor = self.db.posts.find({})
        if limit:
            posts = list(cursor.limit(limit))  # For backtest mode
        else:
            posts = list(cursor)  # Get all posts at once
            
        print(f"Retrieved {len(posts)} posts from database")
        enriched_posts = []

        for post in posts:
            # Resolve references to related entities
            post["contexts"] = self.resolve_reference("contexts", post.get("contexts", []))
            post["countries"] = self.resolve_reference("countries", post.get("countries", []))
            post["primaryCompanies"] = self.resolve_reference("companies", post.get("primaryCompanies", []))
            post["secondaryCompanies"] = self.resolve_reference("companies", post.get("secondaryCompanies", []))
            
            # Handle source reference safely
            source_refs = []
            if post.get("source") and ObjectId.is_valid(post.get("source")):
                source_refs = self.resolve_reference("sources", [post["source"]])
            post["source"] = source_refs[0] if source_refs else None

            # Scrape article from source URL
            if "sourceUrl" in post and post["sourceUrl"]:
                post["scrapedArticle"] = self.scrape_article(post["sourceUrl"])
            else:
                post["scrapedArticle"] = {}
                
            # Process company data with careful checks
            post["primarycompanydata"] = {}
            if (post.get("primaryCompanies") and 
                isinstance(post["primaryCompanies"], list) and 
                len(post["primaryCompanies"]) > 0 and
                isinstance(post["primaryCompanies"][0], dict) and
                "website" in post["primaryCompanies"][0]):
                post["primarycompanydata"] = self.scrape_article(post["primaryCompanies"][0]["website"])
                
            post["secondarycompanydata"] = {}
            if (post.get("secondaryCompanies") and 
                isinstance(post["secondaryCompanies"], list) and 
                len(post["secondaryCompanies"]) > 0 and
                isinstance(post["secondaryCompanies"][0], dict) and
                "website" in post["secondaryCompanies"][0]):
                post["secondarycompanydata"] = self.scrape_article(post["secondaryCompanies"][0]["website"])
                
            enriched_posts.append(post)

        return enriched_posts

    def vectorize_and_upsert(self, post):
        title = post.get("postTitle", "")
        summary = post.get("summary", "")
        contexts = " ".join([c.get("name", "") for c in post.get("contexts", [])])
        article_text = post.get("scrapedArticle", {}).get("text", "")
        
        clean = self.clean_text(" ".join([title, contexts, summary, article_text]))
        if not clean: return

        chunks = self.splitter.create_documents([clean])
        vectors = self.embedding.embed_documents([c.page_content for c in chunks])
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "postId": str(post.get("_id", "")),
                    "postTitle": title,
                    "chunk": chunk.page_content,
                    "chunkIndex": i,
                    "sourceUrl": post.get("sourceUrl", ""),\
                    "summary": summary,
                    "primarydata": post.get("primarycompanydata", {}),
                    "secondarydata": post.get("secondarycompanydata", {}),
                    "contexts": [c.get("name", "") for c in post.get("contexts", [])]
                }
            ) for i, chunk in enumerate(chunks)
        ]
        self.qdrant_client.upsert(collection_name=QdrantConfig.QDRANT_COLLECTION, points=points)
    def search_similar(self, query_text, limit=7):
        """Search for similar documents using vector similarity"""
        try:
            query_vector = self.embedding.embed_query(query_text)
            results = self.qdrant_client.search(
                collection_name=QdrantConfig.QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=limit
            )
            for result in results:
                print("=" * 60)
                print(f"Title: {result.payload.get('postTitle')}")
                print(f"Chunk Index: {result.payload.get('chunkIndex')}")
                print(f"Contexts: {result.payload.get('contexts')}")
                print(f"URL: {result.payload.get('sourceUrl')}")
                print(f"Score: {result.score:.4f}")
                print(f"Content:\n{result.payload.get('chunk')}\n")
        except Exception as e:
            print(f"[Error searching] {str(e)}")


if __name__ == "__main__":
    handler = QdrantHandler()
    
    # complete vector and upsert for all posts
    enriched_posts = handler.get_all_enriched_posts()
    for post in enriched_posts:
        handler.vectorize_and_upsert(post)
    print("All posts vectorized and upserted successfully.")