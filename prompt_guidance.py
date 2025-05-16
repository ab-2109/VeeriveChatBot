import os
import re
import uuid
from typing import List, Dict, Any, Optional, Union
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from qdrant_client import QdrantClient, models
from langchain_community.embeddings import OpenAIEmbeddings
import urllib.parse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('prompt_guidance')

load_dotenv()

class QdrantConfig:
    # Get environment variables with fallbacks
    MONGO_USERNAME = os.getenv("MONGO_USERNAME") or os.getenv("MONGODB_USERNAME", "chaubeyp")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD") or os.getenv("MONGODB_PASSWORD", "ConsTrack360")
    
    MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{urllib.parse.quote_plus(MONGO_PASSWORD)}@veerive.tta8g.mongodb.net/"

    
    QDRANT_URL = os.getenv("QDRANT_URL", "https://9c4151fc-4aaf-418b-ac17-970854ac8a8f.europe-west3-0.gcp.cloud.qdrant.io:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RYAYHgpWnTy9SDZEkpER_1O_QSrvfZ-XTcrq8Wdhkx4")
    QDRANT_COLLECTION = "prompt-guidance"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS = 3072

class PromptGuidanceHandler:
    def __init__(self):
        self.mongo = MongoClient(QdrantConfig.MONGO_URI)["veerive-db"]
        self.qdrant = QdrantClient(url=QdrantConfig.QDRANT_URL, api_key=QdrantConfig.QDRANT_API_KEY)
        self.embedder = OpenAIEmbeddings(model=QdrantConfig.EMBEDDING_MODEL, openai_api_key=QdrantConfig.OPENAI_API_KEY)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if QdrantConfig.QDRANT_COLLECTION not in existing:
            self.qdrant.recreate_collection(
                collection_name=QdrantConfig.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=QdrantConfig.EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {QdrantConfig.QDRANT_COLLECTION}")

    def clean(self, txt: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", txt or "")).strip()

    def resolve_reference(self, collection_name: str, object_id: Union[str, ObjectId, None]) -> Optional[Dict]:
        """Resolve MongoDB reference to actual object - similar to qdbtest.py and qdrant.py"""
        if not object_id:
            return None
            
        # Convert string to ObjectId if needed
        if isinstance(object_id, str) and ObjectId.is_valid(object_id):
            object_id = ObjectId(object_id)
        elif not isinstance(object_id, ObjectId):
            return None
            
        try:
            doc = self.mongo[collection_name].find_one({"_id": object_id})
            return doc
        except Exception as e:
            logger.error(f"Error resolving reference in {collection_name}: {e}")
            return None

    def fetch_documents(self) -> List[Dict[str, Any]]:
        documents = []

        for doc in self.mongo["query_refiner"].find({}):
            try:
                # Get sector and subsector IDs
                sector_id = doc.get("sector")
                print("sector_id", sector_id)
                subsector_id = doc.get("subSector")
                print("subsector_id", subsector_id)
                
                # Resolve references properly
                sector_doc = self.resolve_reference("sectors", sector_id)
                # print("sector_doc", sector_doc)
                subsector_doc = self.resolve_reference("subsectors", subsector_id)
                # print("subsector_doc", subsector_doc)
                
                # Get names or use empty strings
                sector_name = sector_doc.get("sectorName", "") if sector_doc else ""
                subsector_name = subsector_doc.get("subSectorName", "") if subsector_doc else ""
                
                documents.append({
                    "id": str(doc.get("_id", "")),
                    "title": doc.get("title", ""),
                    "prompt": doc.get("promptGuidance", ""),
                    "sector": sector_name,
                    "subsector": subsector_name,
                    "sector_id": str(sector_id) if sector_id else "",
                    "subsector_id": str(subsector_id) if subsector_id else "",
                    "createdAt": doc.get("createdAt"),
                    "updatedAt": doc.get("updatedAt")
                })
                logger.info(f"Processed document: {doc.get('title', 'Untitled')} | Sector: {sector_name}")
                
            except Exception as e:
                logger.error(f"Skipping document {doc.get('_id', 'unknown')} due to error: {e}")
                continue

        return documents

    def vectorize_and_upsert(self, doc: Dict[str, Any]):
        try:
            title = doc.get("title", "")
            raw_prompt = doc.get("prompt", "")
            prompt = self.clean(raw_prompt)
            sector_name = doc.get("sector", "")
            subsector_name = doc.get("subsector", "")

            logger.info(f"Processing: {title}")
            logger.info(f"Sector: {sector_name} | Subsector: {subsector_name}")
            
            if not prompt or not sector_name:
                logger.warning(f"[Skip] Empty prompt or sector for: {title}")
                return

            # Compose weighted query
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

            # Generate a stable ID based on the document's ID if available, or create a new one
            point_id = doc.get("id", uuid.uuid4().hex)
            
            self.qdrant.upsert(
                collection_name=QdrantConfig.QDRANT_COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            logger.info(f"[Upserted] {title} | {sector_name} > {subsector_name}")
        except Exception as e:
            logger.error(f"[Error] {doc.get('id', 'unknown')} => {str(e)}")

    def run(self):
        logger.info("Starting prompt guidance processing")
        documents = self.fetch_documents()
        logger.info(f"Found {len(documents)} documents to process")
        
        for doc in documents:
            self.vectorize_and_upsert(doc)
        
        logger.info("✅ All prompts upserted successfully.")

if __name__ == "__main__":
    handler = PromptGuidanceHandler()
    handler.run()
