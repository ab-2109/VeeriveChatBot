from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from pymongo import MongoClient
import os
import hashlib
import urllib.parse
import time
import logging
import traceback
import schedule
from typing import Dict, Any, List, Set, Optional
from datetime import datetime, timedelta
from bson import ObjectId
from bson.json_util import dumps
import json
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain.schema import Document as LangchainDocument
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("db_sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_synchronizer")

# Load environment variables
load_dotenv()

# Database connection variables
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "chaubeyp")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "ConsTrack360")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "veerive.tta8g.mongodb.net")
MONGO_DB = os.getenv("MONGO_DB", "veerive-db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "posts")

QDRANT_URL = os.getenv("QDRANT_URL", "https://9c4151fc-4aaf-418b-ac17-970854ac8a8f.europe-west3-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RYAYHgpWnTy9SDZEkpER_1O_QSrvfZ-XTcrq8Wdhkx4")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "tester2")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# Sync configuration
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL_MINUTES", "60"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
SYNC_LOOKBACK_DAYS = int(os.getenv("SYNC_LOOKBACK_DAYS", "7"))

class DatabaseSynchronizer:
    def __init__(self):
        self.mongo_client = None
        self.qdrant_client = None
        self.neo4j_driver = None
        self.neo4j_graph = None
        self.embedder = None
        self.llm = None
        self.doc_transformer = None
        self.last_sync_time = None
        self.chrome_options = None
        self.setup_browser_options()
        
    def setup_browser_options(self):
        """Set up Chrome options for web scraping"""
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
        
    def connect(self):
        """Establish connections to all databases"""
        try:
            # Connect to MongoDB
            password = urllib.parse.quote_plus(MONGO_PASSWORD)
            mongo_uri = f"mongodb+srv://{MONGO_USERNAME}:{password}@{MONGO_CLUSTER}/"
            self.mongo_client = MongoClient(mongo_uri)
            self.db = self.mongo_client[MONGO_DB]
            logger.info("Connected to MongoDB")
            
            # Connect to Qdrant
            self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            logger.info("Connected to Qdrant")
            
            # Create collection if it doesn't exist
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            if QDRANT_COLLECTION not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE)  # text-embedding-3-large has 3072 dimensions
                )
                logger.info(f"Created Qdrant collection: {QDRANT_COLLECTION}")
            
            # Connect to Neo4j - standard driver
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            logger.info("Connected to Neo4j driver")
            
            # Connect to Neo4j - graph interface for LangChain
            self.neo4j_graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD
            )
            logger.info("Connected to Neo4j graph")
            
            # Initialize OpenAI embeddings
            self.embedder = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
            logger.info(f"Initialized embeddings model: {EMBED_MODEL}")
            
            # Initialize LLM for graph transformation
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                api_key=OPENAI_API_KEY
            )
            logger.info("Initialized LLM for graph transformations")
            
            # Initialize graph transformer
            self.doc_transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=[
                    "Company", "Product", "Contextsector",  # Fixed casing
                    "Sector", "Subsector", "Location", "Country", 
                    "Signal", "Trend", "News", "Consumer",
                ],
                allowed_relationships=[
                    "LOCATED_IN", "HAS_PRIMARY_COMPANY", "HAS_SECONDARY_COMPANY", 
                    "HAS_SECTOR", "HAS_SUBSECTOR", "HAS_CONTEXT", "HAS_TAG", 
                    "SIGNALS", "PROVIDES_NEWS", "SERVES", "DRIVES", 
                    "IMPACTS", "SUPPORTS", "MENTIONS", "CONTAINS"
                ],
                strict_mode=True,
                node_properties=True,
                relationship_properties=["type", "description"],
            )
            logger.info("Initialized document transformer")
            
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_existing_ids_in_qdrant(self) -> Set[str]:
        """Get all existing MongoDB IDs already stored in Qdrant"""
        try:
            # Check if the collection exists and has points
            collection_info = self.qdrant_client.get_collection(QDRANT_COLLECTION)
            vector_count = collection_info.vectors_count
            
            if vector_count == 0:
                return set()
            
            # We need to scroll through all points to get all IDs
            existing_ids = set()
            offset = 0
            limit = 100
            
            while True:
                points = self.qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=limit,
                    offset=offset
                )[0]
                
                if not points:
                    break
                    
                # Extract MongoDB IDs from payload
                for point in points:
                    if 'postId' in point.payload:
                        existing_ids.add(point.payload['postId'])
                
                offset += limit
                if len(points) < limit:
                    break
            
            logger.info(f"Found {len(existing_ids)} existing documents in Qdrant")
            return existing_ids
        except Exception as e:
            logger.error(f"Error getting existing IDs from Qdrant: {str(e)}")
            return set()

    def clean_text(self, raw_text):
        """Clean text for embedding"""
        if not raw_text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", raw_text)
        # Replace multiple whitespaces with single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def resolve_reference(self, collection_name, object_ids):
        """Resolve MongoDB references to actual objects"""
        if not object_ids:
            return []
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
            
        # Filter out invalid ObjectIDs
        valid_ids = []
        for oid in object_ids:
            if oid and ObjectId.is_valid(oid):
                valid_ids.append(ObjectId(oid))
        
        if not valid_ids:
            return []
            
        docs = list(self.db[collection_name].find({"_id": {"$in": valid_ids}}))
        
        # Convert ObjectId to string in each document
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        
        return docs

    def scrape_article(self, url):
        """Scrape article content from a URL using Selenium"""
        if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL: {url}")
            return {}
            
        driver = None
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.set_page_load_timeout(30)
            
            logger.info(f"Scraping URL: {url}")
            driver.get(url)
            
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
                    # Try CSS selector or tag name
                    if selector.startswith(".") or selector.startswith("#"):
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    else:
                        elements = driver.find_elements(By.TAG_NAME, selector)
                    
                    if elements:
                        for element in elements:
                            if not element.is_displayed():
                                continue
                            
                            # Remove unwanted elements
                            for tag in ['script', 'style', 'nav', 'header', 'footer', 'iframe']:
                                junk_elements = element.find_elements(By.TAG_NAME, tag)
                                for junk in junk_elements:
                                    driver.execute_script("arguments[0].remove()", junk)
                            
                            element_text = element.text.strip()
                            if element_text:
                                text += element_text + "\n\n"
                        
                        if text:
                            break
                except Exception:
                    continue
            
            # If nothing found, fall back to body
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
            
            # Extract authors and date
            authors = []
            for selector in ["[rel='author']", ".author", ".byline", "[itemprop='author']", ".author-name"]:
                try:
                    for element in driver.find_elements(By.CSS_SELECTOR, selector):
                        author = element.text.strip()
                        if author and author not in authors:
                            authors.append(author)
                except:
                    pass
                    
            published_date = None
            for selector in ["time", "[itemprop='datePublished']", "[property='article:published_time']", ".date"]:
                try:
                    for element in driver.find_elements(By.CSS_SELECTOR, selector):
                        published_date = element.get_attribute("datetime") or element.get_attribute("content") or element.text.strip()
                        if published_date:
                            break
                    if published_date:
                        break
                except:
                    pass
            
            return {
                "title": title,
                "text": text,
                "authors": authors,
                "published_date": published_date
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {}
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

    def get_new_mongo_documents(self, existing_ids: Set[str]) -> List[Dict[str, Any]]:
        """Get documents from MongoDB that aren't already in Qdrant or were updated"""
        try:
            # Define the query based on last sync time
            query = {}
            if self.last_sync_time:
                query['updatedAt'] = {'$gte': self.last_sync_time}
            else:
                # If never synced, get recent documents based on lookback days
                lookback_date = datetime.now() - timedelta(days=SYNC_LOOKBACK_DAYS)
                query['updatedAt'] = {'$gte': lookback_date}
            
            # Get total count first to plan batching
            total_count = self.db[MONGO_COLLECTION].count_documents(query)
            logger.info(f"Found {total_count} total posts in MongoDB matching query")
            
            # Process in batches directly from the cursor to avoid loading everything into memory
            processed_count = 0
            enriched_posts = []
            
            # Use cursor with batch_size for memory efficiency
            cursor = self.db[MONGO_COLLECTION].find(query).batch_size(BATCH_SIZE)
            
            for post in cursor:
                doc_id = str(post['_id'])
                
                # Skip if already in Qdrant and not looking to update
                if doc_id in existing_ids and self.last_sync_time is None:
                    continue
                
                # Enrich with related data
                enriched_post = self.enrich_post(post)
                if enriched_post:
                    enriched_posts.append(enriched_post)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{total_count} documents")
            
            logger.info(f"Found {len(enriched_posts)} new/updated documents to process")
            return enriched_posts
        except Exception as e:
            logger.error(f"Error fetching MongoDB documents: {str(e)}")
            traceback.print_exc()
            return []

    def enrich_post(self, post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enrich a post with related data and scraped content"""
        try:
            # Convert ObjectId to string for easier handling
            post["_id"] = str(post["_id"])
            
            # Resolve references to related entities
            post["contexts"] = self.resolve_reference("contexts", post.get("contexts", []))
            post["countries"] = self.resolve_reference("countries", post.get("countries", []))
            post["signals"] = self.resolve_reference("signals", post.get("signals", []))
            post["subsignals"] = self.resolve_reference("subsignals", post.get("subsignals", []))
            post["primaryCompanies"] = self.resolve_reference("companies", post.get("primaryCompanies", []))
            post["secondaryCompanies"] = self.resolve_reference("companies", post.get("secondaryCompanies", []))
            post["sectors"] = self.resolve_reference("sectors", post.get("sectors", []))
            post["subsectors"] = self.resolve_reference("subsectors", post.get("subsectors", []))
            post["themes"] = self.resolve_reference("themes", post.get("themes", []))
            post["regions"] = self.resolve_reference("regions", post.get("regions", []))
            
            # Handle source reference
            if post.get("source") and ObjectId.is_valid(post.get("source")):
                source_docs = self.resolve_reference("sources", [post["source"]])
                post["source"] = source_docs[0] if source_docs else None
            else:
                post["source"] = None

            # Scrape article from source URL if available
            if post.get("sourceUrl"):
                logger.info(f"Scraping article for post {post['_id']}")
                post["scrapedArticle"] = self.scrape_article(post["sourceUrl"])
            else:
                post["scrapedArticle"] = {}
                
            # Scrape company websites when available
            post["primarycompanydata"] = {}
            if post.get("primaryCompanies") and isinstance(post["primaryCompanies"], list) and len(post["primaryCompanies"]) > 0:
                primary = post["primaryCompanies"][0]
                if isinstance(primary, dict) and primary.get("website"):
                    logger.info(f"Scraping primary company website for post {post['_id']}")
                    post["primarycompanydata"] = self.scrape_article(primary["website"])
                    
            post["secondarycompanydata"] = {}
            if post.get("secondaryCompanies") and isinstance(post["secondaryCompanies"], list) and len(post["secondaryCompanies"]) > 0:
                secondary = post["secondaryCompanies"][0]
                if isinstance(secondary, dict) and secondary.get("website"):
                    logger.info(f"Scraping secondary company website for post {post['_id']}")
                    post["secondarycompanydata"] = self.scrape_article(secondary["website"])
            
            return post
        except Exception as e:
            logger.error(f"Error enriching post {post.get('_id', 'unknown')}: {str(e)}")
            return None
    
    def extract_metadata_from_post(self, doc):
        """Extract structured metadata from a MongoDB post document for traceability and filtering"""
        return {
            "doc_id": doc.get("_id", ""),
            "title": doc.get("postTitle", ""),
            "created_at": str(doc.get("createdAt", "")),
            "source_url": doc.get("sourceUrl", ""),
            "post_type": doc.get("postType", ""),
            "sentiment": doc.get("sentiment", ""),
            "tags": doc.get("tags", {}),
            "countries": [c.get("name", "") for c in doc.get("countries", []) if isinstance(c, dict)],
            "contexts": [c.get("name", "") for c in doc.get("contexts", []) if isinstance(c, dict)],
            "primaryCompanies": [c.get("name", "") for c in doc.get("primaryCompanies", []) if isinstance(c, dict)],
            "secondaryCompanies": [c.get("name", "") for c in doc.get("secondaryCompanies", []) if isinstance(c, dict)],
        }
    
    def vectorize_for_qdrant(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare post for vectorization"""
        # Extract text for embedding
        title = post.get("postTitle", "")
        summary = post.get("summary", "")
        content = post.get("content", "")
        
        # Add context information
        contexts = " ".join([c.get("name", "") for c in post.get("contexts", []) if c and isinstance(c, dict)])
        
        # Add article text if available
        article_text = post.get("scrapedArticle", {}).get("text", "")
        
        # Combine all text for embedding
        combined_text = f"{title} {contexts} {summary} {content} {article_text}"
        clean_text = self.clean_text(combined_text)
        
        if not clean_text:
            logger.warning(f"No content to vectorize for post {post['_id']}")
            return None
            
        # Structure for vectorization
        return {
            "post_id": post["_id"],
            "title": title,
            "text_for_embedding": clean_text,
            "summary": summary,
            "source_url": post.get("sourceUrl", ""),
            "contexts": [c.get("name", "") for c in post.get("contexts", []) if c and isinstance(c, dict)],
            "countries": [c.get("name", "") for c in post.get("countries", []) if c and isinstance(c, dict)],
            "sectors": [c.get("name", "") for c in post.get("sectors", []) if c and isinstance(c, dict)],
            "subsectors": [c.get("name", "") for c in post.get("subsectors", []) if c and isinstance(c, dict)],
            "primary_companies": [c.get("name", "") for c in post.get("primaryCompanies", []) if c and isinstance(c, dict)],
            "secondary_companies": [c.get("name", "") for c in post.get("secondaryCompanies", []) if c and isinstance(c, dict)],
            "created_at": post.get("createdAt", datetime.now()).isoformat() if isinstance(post.get("createdAt"), datetime) else str(post.get("createdAt", "")),
            "updated_at": post.get("updatedAt", datetime.now()).isoformat() if isinstance(post.get("updatedAt"), datetime) else str(post.get("updatedAt", ""))
        }
    
    def update_qdrant(self, documents: List[Dict[str, Any]]) -> bool:
        """Add new documents to Qdrant"""
        if not documents:
            logger.info("No documents to add to Qdrant")
            return True
        
        try:
            # Process in batches
            batches = [documents[i:i+BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]
            for i, batch in enumerate(batches):
                logger.info(f"Processing Qdrant batch {i+1}/{len(batches)} ({len(batch)} documents)")
                
                # Prepare vectorized documents
                vectorized_batch = []
                for doc in batch:
                    vectorized_doc = self.vectorize_for_qdrant(doc)
                    if vectorized_doc:
                        vectorized_batch.append(vectorized_doc)
                
                if not vectorized_batch:
                    logger.warning(f"No valid documents in batch {i+1}")
                    continue
                
                # Generate embeddings
                texts = [doc["text_for_embedding"] for doc in vectorized_batch]
                embeddings = self.embedder.embed_documents(texts)
                
                # Prepare points for Qdrant
                points = []
                for j, doc in enumerate(vectorized_batch):
                    # Create point ID from post_id (deterministic)
                    point_id = int(hashlib.md5(doc["post_id"].encode()).hexdigest(), 16) % (2**63)
                    
                    # Remove embedding text from payload to save space
                    payload = {k: v for k, v in doc.items() if k != "text_for_embedding"}
                    
                    points.append(PointStruct(
                        id=point_id,
                        vector=embeddings[j],
                        payload=payload
                    ))
                
                # Upsert to Qdrant
                self.qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points
                )
                
                logger.info(f"Added batch {i+1}/{len(batches)} to Qdrant ({len(points)} points)")
            
            logger.info(f"Successfully added {len(documents)} documents to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error updating Qdrant: {str(e)}")
            traceback.print_exc()
            return False
    
    def setup_neo4j_schema(self):
        """Set up Neo4j schema and constraints using Neo4jGraph interface"""
        try:
            constraints = [
                # Fix casing to match schema.txt
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE", # Lowercase to match schema
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Contextsector) REQUIRE s.id IS UNIQUE", # Changed from SECTOR
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Country) REQUIRE c.id IS UNIQUE", # Lowercase to match schema
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE", # Lowercase
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE", # Lowercase
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Signal) REQUIRE s.id IS UNIQUE", # Lowercase
                # Add missing node types from schema.txt
                "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subsector) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Consumer) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Trend) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:News) REQUIRE n.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    self.neo4j_graph.query(constraint)
                except Exception as e:
                    logger.warning(f"Error creating constraint: {str(e)}")
                    
            # Create indexes for better performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)",
                "CREATE INDEX IF NOT EXISTS FOR (c:COMPANY) ON (c.name)",
                "CREATE INDEX IF NOT EXISTS FOR (s:SECTOR) ON (s.name)",
                "CREATE INDEX IF NOT EXISTS FOR (c:COUNTRY) ON (c.name)"
            ]
            
            for index in indexes:
                try:
                    self.neo4j_graph.query(index)
                except Exception as e:
                    logger.warning(f"Error creating index: {str(e)}")
                    
            logger.info("Neo4j schema setup complete")
            return True
        except Exception as e:
            logger.error(f"Error setting up Neo4j schema: {str(e)}")
            return False

    def update_neo4j(self, documents: List[Dict[str, Any]]) -> bool:
        """Process documents and update Neo4j knowledge graph"""
        if not documents:
            logger.info("No documents to add to Neo4j")
            return True
        
        try:
            # Setup schema - using Neo4jGraph interface for this
            self.setup_neo4j_schema()
            
            # Process documents in batches
            batches = [documents[i:i+BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]
            for i, batch in enumerate(batches):
                logger.info(f"Processing Neo4j batch {i+1}/{len(batches)} ({len(batch)} documents)")
                
                for doc in batch:
                    try:
                        self.process_document_to_neo4j(doc)
                    except Exception as e:
                        logger.error(f"Error processing document {doc['_id']} to Neo4j: {str(e)}")
                        traceback.print_exc()
                
                logger.info(f"Completed Neo4j batch {i+1}/{len(batches)}")
            
            logger.info(f"Knowledge graph updated with {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Error updating Neo4j: {str(e)}")
            traceback.print_exc()
            return False
    
    def process_document_to_neo4j(self, doc):
        """Process a single document and add it to the Neo4j knowledge graph"""
        doc_id = doc["_id"]
        post_title = doc.get("postTitle", "")
        summary = doc.get("summary", "")
        content = doc.get("content", "")
        tags = doc.get("tags", {})
        created_at = doc.get("createdAt", datetime.now())
        source_url = doc.get("sourceUrl", "")
        
        # Extract text content for entity extraction
        text_content = summary or content
        
        if not text_content:
            logger.warning(f"No text content found for document {doc_id}, skipping")
            return
        
        logger.info(f"Processing document {doc_id} - {post_title}")
        
        # 1. Create Document node with basic properties
        self.neo4j_graph.query("""
            MERGE (d:Document {id: $doc_id})
            SET d.title = $title,
                d.summary = $summary,
                d.created_at = $created_at,
                d.source_url = $source_url
            RETURN d
        """, {
            "doc_id": doc_id,
            "title": post_title,
            "summary": summary,
            "created_at": str(created_at),
            "source_url": source_url
        })
        
        # 2. Create a chunk for the content
        chunk_id = f"chunk_{doc_id}"
        
        self.neo4j_graph.query("""
            MATCH (d:Document {id: $doc_id})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text
            MERGE (c)-[:PART_OF]->(d)
        """, {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": text_content
        })
        
        # 3. Process all relationships from the MongoDB document
        
        # 3.1 Create context/sector relationships
        for context in doc.get("contexts", []):
            if isinstance(context, dict) and "name" in context:
                ctx_name = context["name"]
                ctx_id = str(context.get("_id", f"ctx_{ctx_name}"))
                
                self.neo4j_graph.query("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (ctx:SECTOR {id: $ctx_id})
                    SET ctx.name = $ctx_name
                    MERGE (d)-[:HAS_CONTEXT]->(ctx)
                """, {
                    "doc_id": doc_id, 
                    "ctx_id": ctx_id, 
                    "ctx_name": ctx_name
                })
        
        # 3.2 Create country relationships
        for country in doc.get("countries", []):
            if isinstance(country, dict) and "name" in country:
                country_name = country["name"]
                country_id = str(country.get("_id", f"country_{country_name}"))
                
                self.neo4j_graph.query("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:COUNTRY {id: $country_id})
                    SET c.name = $country_name
                    MERGE (d)-[:LOCATED_IN]->(c)
                """, {
                    "doc_id": doc_id, 
                    "country_id": country_id, 
                    "country_name": country_name
                })
        
        # 3.3 Create company relationships
        # Primary companies
        for company in doc.get("primaryCompanies", []):
            if isinstance(company, dict) and "name" in company:
                company_name = company["name"]
                company_id = str(company.get("_id", f"company_{company_name}"))
                website = company.get("website", "")
                
                self.neo4j_graph.query("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:COMPANY {id: $company_id})
                    SET c.name = $company_name,
                        c.website = $website,
                        c.isPrimary = true
                    MERGE (d)-[:ABOUT]->(c)
                """, {
                    "doc_id": doc_id,
                    "company_id": company_id,
                    "company_name": company_name,
                    "website": website
                })
                
        # Secondary companies
        for company in doc.get("secondaryCompanies", []):
            if isinstance(company, dict) and "name" in company:
                company_name = company["name"]
                company_id = str(company.get("_id", f"company_{company_name}"))
                website = company.get("website", "")
                
                self.neo4j_graph.query("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:COMPANY {id: $company_id})
                    SET c.name = $company_name,
                        c.website = $website,
                        c.isSecondary = true
                    MERGE (d)-[:MENTIONS]->(c)
                """, {
                    "doc_id": doc_id,
                    "company_id": company_id,
                    "company_name": company_name,
                    "website": website
                })
        
        # 3.4 Connect companies to sectors and countries
        # This creates a richer graph by connecting entities
        for company in doc.get("primaryCompanies", []) + doc.get("secondaryCompanies", []):
            if isinstance(company, dict) and "name" in company:
                company_id = str(company.get("_id", f"company_{company['name']}"))
                
                # Connect to sectors
                for context in doc.get("contexts", []):
                    if isinstance(context, dict) and "name" in context:
                        ctx_id = str(context.get("_id", f"ctx_{context['name']}"))
                        
                        self.neo4j_graph.query("""
                            MATCH (c:COMPANY {id: $company_id})
                            MATCH (s:SECTOR {id: $ctx_id})
                            MERGE (c)-[:OPERATES_IN]->(s)
                        """, {"company_id": company_id, "ctx_id": ctx_id})
                
                # Connect to countries
                for country in doc.get("countries", []):
                    if isinstance(country, dict) and "name" in country:
                        country_id = str(country.get("_id", f"country_{country['name']}"))
                        
                        self.neo4j_graph.query("""
                            MATCH (c:COMPANY {id: $company_id})
                            MATCH (co:COUNTRY {id: $country_id})
                            MERGE (c)-[:BASED_IN]->(co)
                        """, {"company_id": company_id, "country_id": country_id})
        
        # 3.5 Create source relationships
        source = doc.get("source")
        if source and isinstance(source, dict) and "name" in source:
            source_name = source["name"]
            source_id = str(source.get("_id", f"source_{source_name}"))
            
            self.neo4j_graph.query("""
                MATCH (d:Document {id: $doc_id})
                MERGE (s:SOURCE {id: $source_id})
                SET s.name = $source_name,
                    s.url = $source_url
                MERGE (d)-[:FROM]->(s)
            """, {
                "doc_id": doc_id, 
                "source_id": source_id, 
                "source_name": source_name, 
                "source_url": source_url
            })
        
        # 3.6 Process tags
        if tags:
            for tag_type, tag_value in tags.items():
                if tag_value:
                    self.neo4j_graph.query("""
                        MATCH (d:Document {id: $doc_id})
                        MERGE (t:TAG {type: $tag_type, value: $tag_value})
                        MERGE (d)-[:HAS_TAG]->(t)
                    """, {"doc_id": doc_id, "tag_type": tag_type, "tag_value": tag_value})
        
        # 4. Use LLM to extract additional entities and relationships
        try:
            # Create LangChain document
            lc_doc = LangchainDocument(
                page_content=text_content,
                metadata={
                    **self.extract_metadata_from_post(doc),
                    "chunk_id": chunk_id
                }
            )
            
            # Generate entities and relationships using LLM
            graph_docs = self.doc_transformer.convert_to_graph_documents([lc_doc])
            
            # Connect document/chunk to extracted entities
            document_node = Node(
                id=doc_id,
                type="Document"
            )
            
            chunk_node = Node(
                id=chunk_id,
                type="Chunk"
            )
            
            # For each extracted graph document
            for graph_doc in graph_docs:
                # Add document and chunk as sources
                graph_doc.nodes.append(document_node)
                graph_doc.nodes.append(chunk_node)
                
                for node in graph_doc.nodes:
                    if node.id == doc_id or node.id == chunk_id:
                        continue
                    
                    # Connect document to entity
                    graph_doc.relationships.append(
                        Relationship(
                            source=document_node,
                            target=node,
                            type="CONTAINS"
                        )
                    )
                    
                    # Connect chunk to entity
                    graph_doc.relationships.append(
                        Relationship(
                            source=chunk_node,
                            target=node,
                            type="MENTIONS"
                        )
                    )
                
                # Add to Neo4j with LangChain graph documents
                self.neo4j_graph.add_graph_documents(graph_docs)
            
            logger.info(f"Added {len(graph_docs)} graph documents with {sum(len(doc.nodes) for doc in graph_docs)} nodes and {sum(len(doc.relationships) for doc in graph_docs)} relationships")
                
        except Exception as e:
            logger.error(f"Error extracting entities from document {doc_id}: {str(e)}")
    
    def process_prompt_guidance(self):
        """Process query refiner documents for prompt guidance - similar to prompt_guidance.py"""
        try:
            logger.info("Starting prompt guidance processing")
            
            # Check if the collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            guidance_collection = "prompt-guidance"
            
            if guidance_collection not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=guidance_collection,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {guidance_collection}")
                
            # Get documents from query_refiner collection
            documents = []
            for doc in self.db["query_refiner"].find({}):
                try:
                    # Get sector and subsector IDs
                    sector_id = doc.get("sector")
                    subsector_id = doc.get("subSector")
                    
                    # Resolve references properly
                    sector_doc = None
                    subsector_doc = None
                    
                    if sector_id and ObjectId.is_valid(sector_id):
                        sector_doc = self.db["sectors"].find_one({"_id": ObjectId(sector_id)})
                    
                    if subsector_id and ObjectId.is_valid(subsector_id):
                        subsector_doc = self.db["subsectors"].find_one({"_id": ObjectId(subsector_id)})
                    
                    # Get names or use empty strings
                    sector_name = sector_doc.get("sectorName", "") if sector_doc else ""
                    subsector_name = subsector_doc.get("subSectorName", "") if subsector_doc else ""
                    
                    documents.append({
                        "id": str(doc.get("_id", "")),
                        "title": doc.get("title", ""),
                        "prompt": doc.get("promptGuidance", ""),
                        "sector": sector_name,
                        "subsector": subsector_name,
                        "createdAt": doc.get("createdAt"),
                        "updatedAt": doc.get("updatedAt")
                    })
                    logger.info(f"Processed guidance document: {doc.get('title', 'Untitled')} | Sector: {sector_name}")
                    
                except Exception as e:
                    logger.error(f"Skipping guidance document {doc.get('_id', 'unknown')} due to error: {e}")
                    continue
            
            logger.info(f"Found {len(documents)} guidance documents to process")
            
            # Process documents
            for doc in documents:
                try:
                    title = doc.get("title", "")
                    raw_prompt = doc.get("prompt", "")
                    prompt = self.clean_text(raw_prompt)
                    sector_name = doc.get("sector", "")
                    subsector_name = doc.get("subsector", "")
                    
                    logger.info(f"Processing guidance: {title}")
                    
                    if not prompt or not sector_name:
                        logger.warning(f"[Skip] Empty prompt or sector for: {title}")
                        continue
                    
                    # Compose text for embedding
                    combined_text = f"{sector_name} {subsector_name} {prompt}".strip()
                    embedding = self.embedder.embed_query(combined_text)
                    
                    # Create payload
                    payload = {
                        "id": doc.get("id", ""),
                        "title": title,
                        "prompt": prompt,
                        "sector": sector_name,
                        "subsector": subsector_name,
                        "createdAt": str(doc.get("createdAt")),
                        "updatedAt": str(doc.get("updatedAt")),
                    }
                    
                    # Generate a stable ID based on the document's ID
                    point_id = int(hashlib.md5(doc.get("id", "").encode()).hexdigest(), 16) % (2**63)
                    
                    # Upsert to Qdrant
                    self.qdrant_client.upsert(
                        collection_name=guidance_collection,
                        points=[
                            PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload=payload
                            )
                        ]
                    )
                    logger.info(f"[Upserted] {title} | {sector_name} > {subsector_name}")
                except Exception as e:
                    logger.error(f"[Error] {doc.get('id', 'unknown')} => {str(e)}")
            
            logger.info("✅ All prompt guidance documents processed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error processing prompt guidance documents: {str(e)}")
            return False
    
    def sync(self) -> bool:
        """Main synchronization method"""
        logger.info("Starting database synchronization")
        
        try:
            # Connect to all databases
            if not self.connect():
                logger.error("Failed to connect to one or more databases")
                return False
            
            # Get existing IDs from Qdrant
            existing_ids = self.get_existing_ids_in_qdrant()
            
            # Get new documents from MongoDB
            new_docs = self.get_new_mongo_documents(existing_ids)
            
            success = True
            
            if new_docs:
                logger.info(f"Found {len(new_docs)} new/updated documents to process")
                
                # Update Qdrant
                qdrant_success = self.update_qdrant(new_docs)
                
                # Update Neo4j
                neo4j_success = self.update_neo4j(new_docs)
                
                success = qdrant_success and neo4j_success
            else:
                logger.info("No new or updated documents found")
            
            # Process prompt guidance documents
            guidance_success = self.process_prompt_guidance()
            success = success and guidance_success
            
            if success:
                logger.info("Synchronization completed successfully")
                self.last_sync_time = datetime.now()
                return True
            else:
                logger.error("Synchronization completed with errors")
                return False
                
        except Exception as e:
            logger.error(f"Synchronization error: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            # Close connections
            if self.neo4j_driver:
                self.neo4j_driver.close()
    
    def run_scheduled_sync(self):
        """Run synchronization with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Starting scheduled sync (attempt {attempt+1}/{MAX_RETRIES})")
                success = self.sync()
                if success:
                    logger.info("Scheduled sync completed successfully")
                    break
                else:
                    logger.warning(f"Scheduled sync failed, attempt {attempt+1}/{MAX_RETRIES}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(60)  # Wait before retry
            except Exception as e:
                logger.error(f"Error in scheduled sync: {str(e)}")
                traceback.print_exc()
                if attempt < MAX_RETRIES - 1:
                    time.sleep(60)  # Wait before retry

def main():
    """Main function to run the synchronizer"""
    synchronizer = DatabaseSynchronizer()
    
    # Run initial sync
    logger.info("Running initial synchronization")
    synchronizer.run_scheduled_sync()
    
    # Schedule periodic syncs
    schedule.every(SYNC_INTERVAL).minutes.do(synchronizer.run_scheduled_sync)
    logger.info(f"Scheduled synchronization every {SYNC_INTERVAL} minutes")
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Synchronizer stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        traceback.print_exc()

