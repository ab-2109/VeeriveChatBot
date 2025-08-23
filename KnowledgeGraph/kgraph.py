import os
import json
import neo4j
import urllib.parse
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from bson.objectid import ObjectId

# Database clients
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# LangChain imports
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain.schema import Document as LangchainDocument

from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('knowledge_graph')

# Load environment variables
load_dotenv()

# === DATABASE CONNECTION SETUP ===

# MongoDB connection
MONGODB_USERNAME = os.getenv("MONGO_USERNAME", "chaubeyp")
MONGODB_PASSWORD = os.getenv("MONGO_PASSWORD", "ConsTrack360")
MONGODB_URI = f"mongodb+srv://{MONGODB_USERNAME}:{urllib.parse.quote_plus(MONGODB_PASSWORD)}@veerive.tta8g.mongodb.net/"
mongo_client = MongoClient(MONGODB_URI)
mongodb = mongo_client["veerive-db"]

# Qdrant connection
QDRANT_URL = os.getenv("QDRANT_URL", "https://9c4151fc-4aaf-418b-ac17-970854ac8a8f.europe-west3-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RYAYHgpWnTy9SDZEkpER_1O_QSrvfZ-XTcrq8Wdhkx4")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "tester2")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

try:
    neo4j_driver = neo4j.GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    logger.info(f"Successfully connected to Neo4j at {NEO4J_URI}")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {str(e)}")
    raise

# Initialize Neo4j graph
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Initialize LLM with OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create graph transformer for entity extraction
doc_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[
        "COMPANY", "PRODUCT", "CONTEXT", "CONCEPT",
        "SECTOR", "SUBSECTOR", "LOCATION", "COUNTRY", 
        "SIGNAL", "TREND", "NEWS", "CONSUMER", "TECHNOLOGY",
        "ORGANIZATION", "METRIC", "EVENT",
    ],
    allowed_relationships=[
        "LOCATED_IN", "HAS_PRIMARY_COMPANY", "HAS_SECONDARY_COMPANY", 
        "HAS_SECTOR", "HAS_SUBSECTOR", "HAS_CONTEXT", "HAS_TAG",
        "SIGNALS", "PROVIDES_NEWS", "SERVES", "DRIVES", "IMPACTS", 
        "SUPPORTS", "MENTIONS", "CONTAINS", "OPERATES_IN",
        "BASED_IN", "PART_OF", "ABOUT", "FROM", "RELATED_TO"
    ],
    strict_mode=True,
    node_properties=True,
    relationship_properties=["type", "description"],
)

# === HELPER FUNCTIONS ===

def resolve_references(collection_name: str, object_ids: List) -> List[Dict[str, Any]]:
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
        
    docs = list(mongodb[collection_name].find({"_id": {"$in": valid_ids}}))
    
    # Convert ObjectId to string in each document
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    
    return docs

def extract_metadata_from_post(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured metadata from a MongoDB post document for traceability and filtering"""
    return {
        "doc_id": doc.get("_id", ""),
        "title": doc.get("postTitle", ""),
        "created_at": str(doc.get("createdAt", "")),
        "source_url": doc.get("sourceUrl", ""),
        "post_type": doc.get("postType", ""),
        "sentiment": doc.get("sentiment", ""),
        "tags": doc.get("tags", {}),
        "countries": [c.get("name", "") for c in doc.get("countries", [])],
        "contexts": [c.get("name", "") for c in doc.get("contexts", [])],
        "primaryCompanies": [c.get("name", "") for c in doc.get("primaryCompanies", [])],
        "secondaryCompanies": [c.get("name", "") for c in doc.get("secondaryCompanies", [])],
        "completeContent": doc.get("completecontent", ""),
    }

def setup_schema():
    """Set up Neo4j schema with constraints and vector index"""
    try:
        # Create constraints for unique IDs
        graph.query("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE
        """)
        graph.query("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE
        """)
        graph.query("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (c:COMPANY) REQUIRE c.id IS UNIQUE
        """)
        graph.query("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (s:SECTOR) REQUIRE s.id IS UNIQUE
        """)
        graph.query("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (c:COUNTRY) REQUIRE c.id IS UNIQUE
        """)
        graph.query("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (s:SOURCE) REQUIRE s.id IS UNIQUE
        """)

        # Create vector index for semantic search
        graph.query("""
            CREATE VECTOR INDEX chunkVector
            IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 3072,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        
        logger.info("Successfully set up Neo4j schema and constraints")
        
    except Exception as e:
        logger.error(f"Error setting up schema: {str(e)}")
        raise

# === DATA INGESTION FUNCTIONS ===

def fetch_documents_from_mongodb(limit=20, skip=0) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetch documents from MongoDB with rich relationship data"""
    try:
        # Get posts with all fields
        posts = list(mongodb.posts.find({}).skip(skip).limit(limit))
        
        # Enrich posts with related data
        enriched_posts = []
        for post in posts:
            # Convert ObjectIds to strings for easier handling
            post_id = str(post["_id"])
            post["_id"] = post_id
            
            # Resolve references to related entities
            post["contexts"] = resolve_references("contexts", post.get("contexts", []))
            post["countries"] = resolve_references("countries", post.get("countries", []))
            post["signals"] = resolve_references("signals", post.get("signals", []))
            post["subsignals"] = resolve_references("subsignals", post.get("subsignals", []))
            post["primaryCompanies"] = resolve_references("companies", post.get("primaryCompanies", []))
            post["secondaryCompanies"] = resolve_references("companies", post.get("secondaryCompanies", []))
            post["sectors"] = resolve_references("sectors", post.get("sectors", []))
            post["subsectors"] = resolve_references("subsectors", post.get("subsectors", []))
            post["themes"] = resolve_references("themes", post.get("themes", []))
            post["regions"] = resolve_references("regions", post.get("regions", []))
            post["completeContent"] = post.get("completecontent", "")
            
            # Resolve source
            if post.get("source") and ObjectId.is_valid(post.get("source")):
                sources = resolve_references("sources", [post["source"]])
                post["source"] = sources[0] if sources else None
            else:
                post["source"] = None
                
            enriched_posts.append(post)
        
        # Return enriched posts and a flag indicating if more posts are available
        return enriched_posts, (len(posts) == limit)
    except Exception as e:
        logger.error(f"Error fetching MongoDB documents: {str(e)}")
        return [], False

def fetch_data_from_qdrant(collection_name=QDRANT_COLLECTION, limit=20, offset=None) -> Tuple[List, Any]:
    """Fetch data and embeddings from Qdrant collection"""
    try:
        response = qdrant_client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        logger.info(f"Retrieved {len(response[0])} points from Qdrant collection '{collection_name}'")
        return response[0], response[1]  # (points, next_offset)
    except Exception as e:
        logger.error(f"Error retrieving data from Qdrant: {str(e)}")
        return [], None

def process_mongodb_document(doc):
    """Process a single MongoDB document and add it to the knowledge graph"""
    try:
        doc_id = doc["_id"]
        post_title = doc.get("postTitle", "")
        summary = doc.get("summary", "")
        content = doc.get("content", "")
        tags = doc.get("tags", {})
        created_at = doc.get("createdAt", datetime.now())
        source_url = doc.get("sourceUrl", "")
        completeContent = doc.get("completeContent", "")
        
        # Extract text content for entity extraction
        text_content = completeContent

        if not text_content:
            logger.warning(f"No text content found for document {doc_id}, skipping")
            return
        
        logger.info(f"Processing document {doc_id} - {post_title}")
        
        # 1. Create Document node with basic properties
        doc_props = {
            "doc_id": doc_id,
            "title": post_title,
            "summary": summary,
            "created_at": str(created_at),
            "source_url": source_url,
            "completeContent": completeContent,
        }
        
        graph.query("""
            MERGE (d:Document {id: $doc_id})
            SET d.title = $title,
                d.summary = $summary,
                d.created_at = $created_at,
                d.source_url = $source_url
            SET d.completeContent = $completeContent
            RETURN d
        """, doc_props)
        
        # 2. Create a chunk for the content (simplifying as one chunk per document)
        chunk_id = f"chunk_{doc_id}"
        chunk_text = text_content
        
        # Get embedding for the chunk text
        try:
            embedder = OpenAIEmbeddings(model = "text-embedding-3-large" , api_key=os.getenv("OPENAI_API_KEY"))
            embedding = embedder.embed_query(chunk_text)
        except Exception as e:
            logger.error(f"Error generating embedding for {chunk_id}: {str(e)}")
            embedding = None
            
        chunk_props = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": chunk_text,
            "embedding": embedding
        }
        
        # Create chunk
        if embedding:
            graph.query("""
                MATCH (d:Document {id: $doc_id})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text
                MERGE (c)-[:PART_OF]->(d)
                WITH c
                CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
                RETURN c
            """, chunk_props)
        else:
            graph.query("""
                MATCH (d:Document {id: $doc_id})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text
                MERGE (c)-[:PART_OF]->(d)
                RETURN c
            """, {"doc_id": doc_id, "chunk_id": chunk_id, "text": chunk_text})
        
        # 3. Process all relationships from the MongoDB document
        
        # 3.1 Create context relationships
        for context in doc.get("contexts", []):
            if isinstance(context, dict) and "name" in context:
                ctx_name = context["name"]
                ctx_id = context.get("_id", f"ctx_{ctx_name}")
                
                graph.query("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (ctx:SECTOR {id: $ctx_id})
                    SET ctx.name = $ctx_name
                    MERGE (d)-[:HAS_CONTEXT]->(ctx)
                """, {"doc_id": doc_id, "ctx_id": ctx_id, "ctx_name": ctx_name})
        
        # 3.2 Create country relationships
        for country in doc.get("countries", []):
            if isinstance(country, dict) and "name" in country:
                country_name = country["name"]
                country_id = country.get("_id", f"country_{country_name}")
                
                graph.query("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:COUNTRY {id: $country_id})
                    SET c.name = $country_name
                    MERGE (d)-[:LOCATED_IN]->(c)
                """, {"doc_id": doc_id, "country_id": country_id, "country_name": country_name})
        
        # 3.3 Create company relationships
        for company in doc.get("primaryCompanies", []):
            if isinstance(company, dict) and "name" in company:
                company_name = company["name"]
                company_id = company.get("_id", f"company_{company_name}")
                website = company.get("website", "")
                
                graph.query("""
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
                
        for company in doc.get("secondaryCompanies", []):
            if isinstance(company, dict) and "name" in company:
                company_name = company["name"]
                company_id = company.get("_id", f"company_{company_name}")
                website = company.get("website", "")
                
                graph.query("""
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
        
        # 3.4 Create relationships between companies and sectors/countries
        for company in doc.get("primaryCompanies", []) + doc.get("secondaryCompanies", []):
            if isinstance(company, dict) and "name" in company:
                company_id = company.get("_id", f"company_{company['name']}")
                
                # Connect companies to sectors
                for context in doc.get("contexts", []):
                    if isinstance(context, dict) and "name" in context:
                        ctx_id = context.get("_id", f"ctx_{context['name']}")
                        
                        graph.query("""
                            MATCH (c:COMPANY {id: $company_id})
                            MATCH (s:SECTOR {id: $ctx_id})
                            MERGE (c)-[:OPERATES_IN]->(s)
                        """, {"company_id": company_id, "ctx_id": ctx_id})
                
                # Connect companies to countries
                for country in doc.get("countries", []):
                    if isinstance(country, dict) and "name" in country:
                        country_id = country.get("_id", f"country_{country['name']}")
                        
                        graph.query("""
                            MATCH (c:COMPANY {id: $company_id})
                            MATCH (co:COUNTRY {id: $country_id})
                            MERGE (c)-[:BASED_IN]->(co)
                        """, {"company_id": company_id, "country_id": country_id})
        
        # 3.5 Create source relationships
        source = doc.get("source")
        if source and isinstance(source, dict) and "name" in source:
            source_name = source["name"]
            source_id = source.get("_id", f"source_{source_name}")
            
            graph.query("""
                MATCH (d:Document {id: $doc_id})
                MERGE (s:SOURCE {id: $source_id})
                SET s.name = $source_name,
                    s.url = $source_url
                MERGE (d)-[:FROM]->(s)
            """, {"doc_id": doc_id, "source_id": source_id, "source_name": source_name, "source_url": source_url})
                
        # 3.6 Process tags
        if tags:
            for tag_type, tag_value in tags.items():
                if tag_value:
                    graph.query("""
                        MATCH (d:Document {id: $doc_id})
                        MERGE (t:TAG {type: $tag_type, value: $tag_value})
                        MERGE (d)-[:HAS_TAG]->(t)
                    """, {"doc_id": doc_id, "tag_type": tag_type, "tag_value": tag_value})
        
        # 4. Use LangChain to extract additional entities and relationships
        try:
            # Create LangChain document
            lc_doc = LangchainDocument(
                page_content=text_content,
                metadata={
                    **extract_metadata_from_post(doc),
                    "chunk_id": chunk_id
                }
            )
            
            # Generate entities and relationships using LLM
            graph_docs = doc_transformer.convert_to_graph_documents([lc_doc])
            
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
                
                # Add to Neo4j
                graph.add_graph_documents(graph_docs)
                
            logger.info(f"Added {len(graph_docs)} graph documents with {sum(len(doc.nodes) for doc in graph_docs)} nodes and {sum(len(doc.relationships) for doc in graph_docs)} relationships")
                
        except Exception as e:
            logger.error(f"Error processing document {doc_id} with LangChain: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {str(e)}")
        raise

def build_knowledge_graph(backtest=False):
    """Build knowledge graph using MongoDB data sources with optional backtest mode"""
    logger.info("Setting up Neo4j schema...")
    setup_schema()
    
    logger.info("Building knowledge graph from MongoDB documents...")
    batch_size = 20
    skip = 0
    has_more = True
    processed_count = 0
    max_documents = 1000
    
    while has_more and processed_count < max_documents:
        documents, has_more = fetch_documents_from_mongodb(limit=batch_size, skip=skip)
        if not documents:
            logger.info("No more documents to process from MongoDB")
            break
            
        for doc in documents:
            process_mongodb_document(doc)
            processed_count += 1
            
            if processed_count >= max_documents:
                break
                
        skip += batch_size
        logger.info(f"Processed {processed_count} documents so far")
    
    logger.info("Knowledge graph creation complete!")
    
    # Get and display statistics
    result = graph.query("MATCH (n) RETURN labels(n)[0] as label, count(*) as count")
    logger.info("Graph Statistics:")
    for row in result:
        logger.info(f"- {row['label']}: {row['count']} nodes")
    
    result = graph.query("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
    for row in result:
        logger.info(f"- {row['type']}: {row['count']} relationships")

def query_graph(query_text):
    """Query the graph using a text query"""
    # Get all companies that match the query text
    results = graph.query("""
        MATCH (c:COMPANY)
        WHERE c.name CONTAINS $query_text
        RETURN c.id as company_id, c.name as company_name, c.website as website
        LIMIT 5
    """, {"query_text": query_text})
    
    # If no direct matches, try to find related concepts
    if not results:
        results = graph.query("""
            MATCH (n)
            WHERE (n:COMPANY OR n:SECTOR OR n:CONCEPT) AND n.name CONTAINS $query_text
            RETURN n.id as entity_id, n.name as entity_name, labels(n)[0] as entity_type
            LIMIT 5
        """, {"query_text": query_text})
    
    return results

def sample_knowledge_query(entity_name, limit=10):
    """Run a sample query to show connected entities for a given entity"""
    results = graph.query(f"""
        MATCH (n)-[r]-(m)
        WHERE n.name = $entity_name OR n.id = $entity_name
        RETURN n.name as source, labels(n)[0] as source_type, 
               type(r) as relationship, 
               m.name as target, labels(m)[0] as target_type, 
               m.id as target_id
        LIMIT {limit}
    """, {"entity_name": entity_name})
    
    logger.info(f"Connections for '{entity_name}':")
    for row in results:
        logger.info(f"- {row['source']} ({row['source_type']}) --[{row['relationship']}]--> {row['target']} ({row['target_type']})")
    
    return results

def run_business_queries():
    """Run some business-focused queries on the knowledge graph"""
    # Companies by sector
    logger.info("Companies by Sector:")
    results = graph.query("""
        MATCH (c:COMPANY)-[:OPERATES_IN]->(s:SECTOR)
        RETURN s.name as sector, collect(c.name) as companies, count(c) as company_count
        ORDER BY company_count DESC
        LIMIT 10
    """)
    for row in results:
        logger.info(f"- {row['sector']}: {row['company_count']} companies")
        logger.info(f"  {', '.join(row['companies'][:3])}{'...' if len(row['companies']) > 3 else ''}")
    
    # Top countries by company presence
    logger.info("Top Countries by Company Presence:")
    results = graph.query("""
        MATCH (c:COMPANY)-[:BASED_IN]->(country:COUNTRY)
        RETURN country.name as country, count(c) as company_count
        ORDER BY company_count DESC
        LIMIT 5
    """)
    for row in results:
        logger.info(f"- {row['country']}: {row['company_count']} companies")

def main():
    """Main entry point for the knowledge graph module"""
    build_knowledge_graph(backtest=False)

if __name__ == "__main__":
    main()

# === PUBLIC API ===

def ingest_documents(docs):
    """Public entry point: ingest a list of MongoDB post documents into the knowledge graph.
    Expects documents already enriched with resolved refs (contexts, countries, companies, etc.).
    """
    if not docs:
        logger.info("ingest_documents: nothing to ingest")
        return {"ingested": 0}
    count = 0
    for d in docs:
        try:
            process_mongodb_document(d)
            count += 1
        except Exception as e:
            logger.error(f"Failed to ingest doc {d.get('_id','unknown')}: {e}")
    logger.info(f"ingest_documents: ingested {count} documents into KG")
    return {"ingested": count}