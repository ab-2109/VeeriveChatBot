from pymongo import MongoClient
from bson.objectid import ObjectId
import urllib.parse
import pprint
import uuid
import time
import os
from dotenv import load_dotenv
# Replace SentenceTransformer with OpenAI
from openai import OpenAI
from qdrant_client import QdrantClient, models
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

# --- Configuration ---
class QdrantConfig:
    # MongoDB credentials
    MONGODB_USERNAME = "chaubeyp"
    MONGODB_PASSWORD = "ConsTrack360"
    MONGODB_URI = f"mongodb+srv://{MONGODB_USERNAME}:{urllib.parse.quote_plus(MONGODB_PASSWORD)}@veerive.tta8g.mongodb.net/"
    
    # Qdrant credentials
    QDRANT_URL = "https://9c4151fc-4aaf-418b-ac17-970854ac8a8f.europe-west3-0.gcp.cloud.qdrant.io:6333"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RYAYHgpWnTy9SDZEkpER_1O_QSrvfZ-XTcrq8Wdhkx4"
    QDRANT_COLLECTION = "tester1"
    
    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large has 3072 dimensions
    
    # Target collections for RAG/knowledge graph
    TARGET_COLLECTIONS = [
        "posts", "signals", "subsignals", "sources", "sectors", "subsectors",
        "companies", "themes", "countries", "contexts", "regions"
    ]


class QdrantHandler:
    def __init__(self):
        # Initialize MongoDB client
        self.mongo_client = MongoClient(QdrantConfig.MONGODB_URI)
        self.db = self.mongo_client["veerive-db"]
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=QdrantConfig.QDRANT_URL,
            api_key=QdrantConfig.QDRANT_API_KEY
        )
        
        # Initialize OpenAI client instead of SentenceTransformer
        self.openai_client = OpenAI(api_key=QdrantConfig.OPENAI_API_KEY)
        
        # Ensure collection exists with correct vector dimensions
        self._ensure_collection_exists()
        
        # Initialize browser options for Selenium
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        
        # Set browser preferences to block unnecessary content
        prefs = {
            "profile.managed_default_content_settings.images": 2,  # Block images
            "profile.managed_default_content_settings.javascript": 1,  # Allow JS
            "profile.default_content_setting_values.notifications": 2,  # Block notifications
            "profile.managed_default_content_settings.plugins": 2,  # Block plugins
        }
        self.chrome_options.add_experimental_option("prefs", prefs)
    
    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with correct vector dimensions"""
        collection_name = QdrantConfig.QDRANT_COLLECTION
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            print(f"Creating collection: {collection_name}")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=QdrantConfig.EMBEDDING_DIMENSIONS,
                    distance=models.Distance.COSINE
                )
            )
    
    def generate_embedding(self, text):
        """Generate embedding using OpenAI's text-embedding-3-large model"""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
            
        # Truncate text if too long (OpenAI has token limits)
        if len(text) > 8000:  # Approximate character limit
            text = text[:8000]
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=QdrantConfig.EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise
    
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        if hasattr(self, 'mongo_client') and self.mongo_client:
            self.mongo_client.close()
    
    # --- Helper Methods ---
    def resolve_reference(self, collection_name, object_ids):
        """Resolve MongoDB references to actual objects"""
        if not object_ids:
            return []
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
            
        # Filter out invalid ObjectIDs
        valid_ids = [ObjectId(oid) for oid in object_ids if ObjectId.is_valid(oid) if oid]
        if not valid_ids:
            return []
            
        docs = self.db[collection_name].find({"_id": {"$in": valid_ids}})
        return list(docs)
    
    def scrape_article(self, url):
        """Scrape article content from a URL using Selenium"""
        # Validate URL scheme
        if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            print(f"[Invalid URL] {url}")
            return {}
            
        driver = None
        try:
            # Initialize the WebDriver
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.set_page_load_timeout(30)  # Set page load timeout
            
            # Navigate to the URL
            print(f"Scraping: {url}")
            driver.get(url)
            
            # Wait for page to load (wait for body element)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract title
            title = driver.title
            
            # Try to find main content container using common selectors
            content_selectors = [
                "article", "main", ".article-content", ".post-content", 
                "#content", ".content", ".entry-content", ".article-body"
            ]
            
            text = ""
            for selector in content_selectors:
                try:
                    # Try CSS selector first
                    if selector.startswith(".") or selector.startswith("#"):
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    else:
                        elements = driver.find_elements(By.TAG_NAME, selector)
                    
                    if elements:
                        for element in elements:
                            # Skip hidden elements
                            if not element.is_displayed():
                                continue
                            
                            # Clear unnecessary elements
                            for tag in ['script', 'style', 'nav', 'header', 'footer', 'iframe']:
                                junk_elements = element.find_elements(By.TAG_NAME, tag)
                                for junk in junk_elements:
                                    driver.execute_script("arguments[0].remove()", junk)
                            
                            element_text = element.text.strip()
                            if element_text:
                                text += element_text + "\n\n"
                        
                        if text:
                            break  # Stop if we found content
                except Exception as e:
                    continue  # Try next selector
            
            # If we couldn't find content with specific selectors, fall back to body
            if not text:
                try:
                    body = driver.find_element(By.TAG_NAME, "body")
                    
                    # Remove noisy elements
                    for tag in ['script', 'style', 'nav', 'header', 'footer', 'iframe']:
                        junk_elements = body.find_elements(By.TAG_NAME, tag)
                        for junk in junk_elements:
                            driver.execute_script("arguments[0].remove()", junk)
                            
                    text = body.text
                except:
                    text = ""
            
            # Extract authors
            authors = []
            author_selectors = [
                "[rel='author']", ".author", ".byline", "[itemprop='author']",
                ".author-name", ".writer", ".contributor"
            ]
            
            for selector in author_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        author_text = element.text.strip()
                        if author_text and author_text not in authors:
                            authors.append(author_text)
                except:
                    continue
            
            # Extract publication date
            published_date = None
            date_selectors = [
                "time", "[itemprop='datePublished']", "[property='article:published_time']", 
                ".date", ".published-date", ".publish-date"
            ]
            
            for selector in date_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        # Check for datetime attribute
                        date_attr = element.get_attribute("datetime")
                        if date_attr:
                            published_date = date_attr
                            break
                        
                        # Check for content attribute
                        date_content = element.get_attribute("content")
                        if date_content:
                            published_date = date_content
                            break
                        
                        # Fall back to text content
                        date_text = element.text.strip()
                        if date_text:
                            published_date = date_text
                            break
                    
                    if published_date:
                        break
                except:
                    continue
            
            # Clean up and close browser
            driver.quit()
            
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
                try:
                    driver.quit()
                except:
                    pass
    
    # --- Main Processing Methods ---
    def get_all_enriched_posts(self):
        """Fetch posts and resolve all references to create enriched documents"""
        posts = self.db.posts.find({})
        enriched_posts = []

        for post in posts:
            # Resolve references to related entities
            post["contexts"] = self.resolve_reference("contexts", post.get("contexts", []))
            post["countries"] = self.resolve_reference("countries", post.get("countries", []))
            post["primaryCompanies"] = self.resolve_reference("companies", post.get("primaryCompanies", []))
            post["secondaryCompanies"] = self.resolve_reference("companies", post.get("secondaryCompanies", []))
            
            # Handle source reference
            if post.get("source") and ObjectId.is_valid(post.get("source")):
                source_docs = self.resolve_reference("sources", [post["source"]])
                post["source"] = source_docs[0] if source_docs else None
            else:
                post["source"] = None

            # Scrape article from source URL
            if "sourceUrl" in post and post["sourceUrl"]:
                post["scrapedArticle"] = self.scrape_article(post["sourceUrl"])
            else:
                post["scrapedArticle"] = {}
                
            # Scrape company website data when available
            if 'website' in post['primaryCompanies'][0] :
                primary_company = post["primaryCompanies"][0]
                if primary_company.get("website"):
                    post["primarycompanydata"] = self.scrape_article(primary_company["website"])
                else:
                    post["primarycompanydata"] = {}
            else:
                post["primarycompanydata"] = {}
                
            if 'website' in post['secondaryCompanies'][0] :
                secondary_company = post["secondaryCompanies"][0]
                if secondary_company.get("website"):
                    post["secondarycompanydata"] = self.scrape_article(secondary_company["website"])
                else:
                    post["secondarycompanydata"] = {}
            else:
                post["secondarycompanydata"] = {}

            enriched_posts.append(post)

        return enriched_posts
    
    def vectorize_and_upsert(self, post):
        """Generate embeddings for a post and store in Qdrant"""
        # Extract text fields for embedding
        post_title = post.get("postTitle", "")
        summary = post.get("summary", "")
        
        # Join context names if available
        contexts = " ".join([c.get("name", "") for c in post.get("contexts", []) if c])
        
        # Get article text if available
        article_text = post.get("scrapedArticle", {}).get("text", "")

        # Combine all text for embedding
        text_to_embed = " ".join([post_title, contexts, summary, article_text]).strip()
        if not text_to_embed:
            print(f"[Skip] Empty content for post '{post_title}'")
            return

        try:
            # Generate embedding using OpenAI instead of SentenceTransformer
            embedding = self.generate_embedding(text_to_embed)
            
            # Create metadata payload
            payload = {
                "postId": str(post.get("_id", "")),
                "postTitle": post_title,
                "contexts": [c.get("name", "") for c in post.get("contexts", []) if c],
                "summary": summary,
                "sourceUrl": post.get("sourceUrl", ""),
                "scrapedArticle": post.get("scrapedArticle", {}),
                "primarycompanydata": post.get("primarycompanydata", {}),
                "secondarycompanydata": post.get("secondarycompanydata", {})
            }

            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=QdrantConfig.QDRANT_COLLECTION,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            print(f"[Upserted] '{post_title}'")
        except Exception as e:
            print(f"[Error vectorizing] '{post_title}': {str(e)}")
    
    def search_similar(self, query_text, limit=5):
        """Search for similar documents using vector similarity"""
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query_text)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=QdrantConfig.QDRANT_COLLECTION,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Extract results
            results = []
            for result in search_results:
                results.append({
                    "postTitle": result.payload.get("postTitle"),
                    "summary": result.payload.get("summary"),
                    "score": result.score,
                    "sourceUrl": result.payload.get("sourceUrl"),
                    "contexts": result.payload.get("contexts", [])
                })
            
            return results
        except Exception as e:
            print(f"[Error searching] {str(e)}")
            return []
    
    def process_all_posts(self):
        """Process all posts: enrich, vectorize, and upsert to Qdrant"""
        enriched_posts = self.get_all_enriched_posts()
        print(f"Found {len(enriched_posts)} posts to process")
        
        for i, post in enumerate(enriched_posts):
            print(f"Processing post {i+1}/{len(enriched_posts)}")
            self.vectorize_and_upsert(post)
    
    def fetch_sample_documents(self, limit=2):
        """Fetch and display sample documents from each collection"""
        for collection_name in QdrantConfig.TARGET_COLLECTIONS:
            collection = self.db[collection_name]
            print(f"\n--- Collection: {collection_name} ---")
            for doc in collection.find().limit(limit):
                pprint.pprint(doc)


# --- Script Execution ---
if __name__ == "__main__":
    handler = QdrantHandler()
    
    # Uncomment the functionality you want to execute
    
    # Option 1: View sample documents from collections
    # handler.fetch_sample_documents(limit=2)
    
    # # Option 2: Process all posts (enrich, vectorize, upsert)
    handler.process_all_posts()
    
    # Option 3: Test similarity search
    # results = handler.search_similar("BNPL companies in India")
    # for result in results:
    #     print(f"Title: {result['postTitle']}")
    #     print(f"Score: {result['score']}")
    #     print(f"URL: {result['sourceUrl']}")
    #     print("-" * 50)

