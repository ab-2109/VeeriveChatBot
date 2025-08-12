from qdrant_client import QdrantClient
from pymongo import MongoClient
from langchain_community.graphs import Neo4jGraph
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import os
import urllib.parse
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retrieval_agent')
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_openai import OpenAIEmbeddings
import os
load_dotenv()

def convert_paths_to_natural_language(path_rows):
    """
    Standalone helper to convert Neo4j path query rows into natural language.
    Exported so agents.__init__ can import it.
    """
    statements = []
    for row in path_rows or []:
        if not isinstance(row, dict):
            continue
        start = row.get("start_name", "Unknown")
        start_type = row.get("start_type", "")
        end = row.get("end_name", "Unknown")
        end_type = row.get("end_type", "")
        rel = row.get("relationship", "related to")

        end_props = row.get("end_properties", {}) or {}
        sentence = f"{start_type} '{start}' {rel.replace('_', ' ').lower()} {end_type} '{end}'."

        if "description" in end_props:
            sentence += f" {end} is described as: {end_props['description']}."
        if "value" in end_props:
            sentence += f" It has a value of {end_props['value']}."
        if "trend" in end_props:
            sentence += f" The trend observed is {end_props['trend']}."
        if "growth" in end_props:
            sentence += f" Observed growth is {end_props['growth']}."

        statements.append(sentence)
    return statements

class PromptSearcher:
    def __init__(self, qdrant_url: str, qdrant_key: str, collection_name: str = "prompt-guidance"):
        """Initialize the PromptSearcher with the correct collection name (hyphen, not underscore)"""
        self.qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_key
        )
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = collection_name

    def rerank_results(self, hits, sector=None, subsector=None):
        """Rerank search results based on sector and subsector matches"""
        reranked = []
        for hit in hits:
            score = hit.score
            payload = hit.payload or {}
            if sector and payload.get("sector") == sector:
                score += 0.2  # boost for sector match
            if subsector and payload.get("subsector") == subsector:
                score += 0.1  # extra boost for subsector match
            reranked.append((score, payload))
        reranked.sort(reverse=True, key=lambda x: x[0])
        return reranked

    def search_prompt(self, query_text: str, sector: str = None, subsector: str = None, top_k: int = 1):
        """Search for relevant prompts with improved error handling"""
        try:
            query_vector = self.embedder.embed_query(query_text)
            
            # Check if collection exists
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection not in collection_names:
                logging.warning(f"Collection '{self.collection}' does not exist in Qdrant")
                return []
                
            results = self.qdrant.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=20,  # fetch more to rerank
                with_payload=True,
                with_vectors=False,
                score_threshold=0.7,
            )

            reranked = self.rerank_results(results, sector, subsector)
            
            # Return format compatible with raggen.py expectations
            return [
                {
                    "title": doc.get("title", ""),
                    "prompt": doc.get("prompt", ""),
                    "sector": doc.get("sector", ""),
                    "subsector": doc.get("subsector", ""),
                    "score": round(score, 4)
                }
                for score, doc in reranked[:top_k] if doc.get("prompt")
            ]
        except Exception as e:
            logging.error(f"Error searching for prompts: {str(e)}")
            return []  # Return empty list on error



class KGReasoner:
    def __init__(self, graph):
        self.graph = graph

    def reason_over_company_relationships(self, company: str):
        # Company → {Product|Concept|Sector|Country|Consumer} via common rels
        q = """
        MATCH path = (c:Company {id: $company})
                     -[:SUPPORTS|SERVES|DRIVES|IMPACTS|OPERATES_IN|HAS_SECTOR|HAS_CONTEXT*1..2]-
                     (e)
        RETURN path,
               c.id AS company,
               labels(e)[0] AS entity_type,
               e.id AS related_entity,
               type(relationships(path)[-1]) AS relationship_type
        LIMIT 10
        """
        return self.graph.query(q, {"company": company})

    def reason_over_sector_trends(self, sector: str):
        # Sector → Concepts and optional News
        q = """
        MATCH (s:Sector {id: $sector})
        OPTIONAL MATCH path = (s)
            -[:DRIVES|IMPACTS|RELATED_TO|HAS_CONTEXT*1..2]- (c:Concept)
        OPTIONAL MATCH (c)-[:PROVIDES_NEWS|MENTIONS]->(n:News)
        RETURN path, s.id AS sector,
               CASE WHEN c IS NULL THEN NULL ELSE c.id END AS concept,
               CASE WHEN n IS NULL THEN NULL ELSE n.id END AS news_id
        LIMIT 10
        """
        return self.graph.query(q, {"sector": sector})

    def reason_market_trends(self, signal: str):
        # Treat "signal" as a Concept
        q = """
        MATCH path = (sig:Concept {id: $signal})
                     -[:IMPACTS|DRIVES|RELATED_TO*1..2]-(e)
        RETURN path, sig.id AS signal,
               labels(e)[0] AS impacted_type,
               e.id AS impacted_entity,
               type(relationships(path)[-1]) AS relationship
        LIMIT 8
        """
        return self.graph.query(q, {"signal": signal})

    def reason_over_product_impact(self, product: str):
        q = """
        MATCH path = (p:Product {id: $product})
                     -[:IMPACTS|DRIVES|SUPPORTS|SERVES|RELATED_TO*1..2]-(e)
        RETURN path, p.id AS product,
               labels(e)[0] AS impacted_type,
               e.id AS impacted_entity,
               type(relationships(path)[-1]) AS relationship_type
        LIMIT 8
        """
        return self.graph.query(q, {"product": product})

    def reason_over_business_model(self, sector: str, country: Optional[str] = None):
        # Sector anchored: Companies in the sector, optionally filtered by country
        q = """
        MATCH (s:Sector {id: $sector})
        MATCH (company:Company)-[:HAS_SECTOR|OPERATES_IN|HAS_CONTEXT]->(s)
        OPTIONAL MATCH (company)-[:LOCATED_IN|BASED_IN]->(co:Country)
        OPTIONAL MATCH (company)-[:SERVES]->(consumer:Consumer)
        OPTIONAL MATCH (company)-[r:DRIVES|IMPACTS]->(concept:Concept)
        WITH company,
             COLLECT(DISTINCT consumer.id) AS consumers,
             COLLECT(DISTINCT {concept: concept.id, rel: type(r), desc: r.description}) AS concepts,
             co
        WHERE $country IS NULL OR (co.id = $country)
        RETURN company.id AS company,
               consumers,
               concepts
        LIMIT 10
        """
        return self.graph.query(q, {"sector": sector, "country": country})

    def reason_document_insights(self, tags: dict):
        # Documents attach via :CONTAINS to Company/Sector/Country/Subsector/Concept etc.
        conditions = []
        params = {}

        if tags.get("company"):
            conditions.append("(d)-[:CONTAINS]->(:Company {id: $company})")
            params["company"] = tags["company"]
        if tags.get("sector"):
            conditions.append("(d)-[:CONTAINS]->(:Sector {id: $sector})")
            params["sector"] = tags["sector"]
        if tags.get("country"):
            conditions.append("(d)-[:CONTAINS]->(:Country {id: $country})")
            params["country"] = tags["country"]
        if tags.get("subsector"):
            conditions.append("(d)-[:CONTAINS]->(:Subsector {id: $subsector})")
            params["subsector"] = tags["subsector"]
        if tags.get("concept"):
            conditions.append("(d)-[:CONTAINS]->(:Concept {id: $concept})")
            params["concept"] = tags["concept"]

        if not conditions:
            return self.graph.query("""
                MATCH (d:Document)
                RETURN d.id AS id, d.title AS title, d.summary AS summary, d.source_url AS url
                LIMIT 5
            """)

        q = f"""
        MATCH (d:Document)
        WHERE {' AND '.join(conditions)}
        RETURN d.id AS id, d.title AS title, d.summary AS summary, d.source_url AS url
        LIMIT 10
        """
        return self.graph.query(q, params)

    def reason(self, tags: dict):
        results = []

        if tags.get("company"):
            results += self.reason_over_company_relationships(tags["company"])

        if tags.get("sector"):
            results += self.reason_over_sector_trends(tags["sector"])
            if tags.get("country"):
                results += self.reason_over_business_model(tags["sector"], tags["country"])

        if tags.get("query_type", "").lower() == "business model" and tags.get("sector"):
            results += self.reason_over_business_model(tags["sector"], tags.get("country"))

        if tags.get("signal"):
            results += self.reason_market_trends(tags["signal"])

        results += self.reason_document_insights(tags)
        return results

class RetrievalAgent:
    def __init__(self, mongo_uri: str, qdrant_url: str, qdrant_key: str,
                 neo4j_uri: str, neo4j_user: str, neo4j_pass: str,
                 qdrant_collection: str = "tester2",
                 pdf_collection: str = "veerive_docs",  
                 embed_model: str = "text-embedding-3-large"):

        # MongoDB
        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_db = self.mongo_client['veerive-db']
        
        # Store references to all target collections
        self.target_collections = {
            name: self.mongo_db[name] for name in [
                "posts", "signals", "subsignals", "sources", "sectors", "subsectors",
                "companies", "themes", "countries", "contexts", "regions"
            ]
        }

        self.mongo_collection = self.mongo_db['posts']
        
        # Qdrant
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        self.qdrant_collection = qdrant_collection
        self.pdf_collection = pdf_collection 

        self.embedder = OpenAIEmbeddings(model=embed_model, api_key=os.getenv("OPENAI_API_KEY"))

        # Neo4j
        self.neo4j_graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pass)
        
        # Initialize the knowledge graph reasoner
        self.kg_reasoner = KGReasoner(self.neo4j_graph)


    def retrieve_from_qdrant(self, query_text: str, top_k: int = 15):
        """Search regular (non‑PDF) content in posts collection."""
        try:
            query_vector = self.embedder.embed_query(query_text)
            hits = self.qdrant_client.search(
                collection_name=self.qdrant_collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                timeout=10,
                score_threshold=0.35,
            )
            logger.info(f"[Qdrant posts] Retrieved {len(hits)} hits")
            
            # Print detailed chunk information
            logger.info("=== QDRANT CHUNKS RETRIEVED ===")
            for i, hit in enumerate(hits):
                payload = hit.payload or {}
                text_preview = payload.get('text', payload.get('content', ''))[:200]
                logger.info(f"Chunk {i+1}:")
                logger.info(f"  Score: {hit.score:.4f}")
                logger.info(f"  PostID: {payload.get('postId', 'N/A')}")
                logger.info(f"  Text preview: {text_preview}...")
                logger.info(f"  Full payload keys: {list(payload.keys())}")
                logger.info("  " + "-" * 50)
            logger.info("=== END QDRANT CHUNKS ===")
            
            return hits
        except Exception as e:
            logger.error(f"[Qdrant posts] Error: {e}")
            return []

    def retrieve_prompt(self, query_text: str, top_k: int = 1):
        """Retrieve prompt guidance from Qdrant based on the query text with better error handling"""
        try:
            QDRANT_URL = os.getenv("QDRANT_URL")
            QDRANT_API = os.getenv("QDRANT_API")
            
            # Get tags from the current query context if available
            tags = getattr(self, 'current_tags', {})
            sector = tags.get('sector')
            subsector = tags.get('subsector')
            
            prompt_searcher = PromptSearcher(QDRANT_URL, QDRANT_API, collection_name="prompt-guidance")
            prompts = prompt_searcher.search_prompt(query_text, sector=sector, subsector=subsector, top_k=top_k)
            
            if not prompts:
                logging.info("No matching prompts found")
                return []
                
            logging.info(f"Found {len(prompts)} matching prompts")
            return prompts
        except Exception as e:
            logging.error(f"Error retrieving prompts: {str(e)}")
            return []  # Return empty list on error

    def retrieve_from_neo4j(self, tags: dict, query_type: str = None):
        """Query Neo4j graph based on tags using actual schema relationships"""
        results = []

        # Company-centric paths
        if "company" in tags and tags["company"]:
            company_queries = [
                # Company to products
                """
                MATCH path = (c:Company {id: $company})
                            -[:SUPPORTS|SERVES|DRIVES]->
                            (p:Product)
                RETURN path, c.id as company, p.id as product,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """,
                # Company to sectors
                """
                MATCH path = (c:Company {id: $company})
                            -[:HAS_CONTEXT|HAS_SECTOR]->
                            (s:Sector)
                RETURN path, c.id as company, s.id as sector,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """,
                # Company to trends
                """
                MATCH path = (c:Company {id: $company})
                            -[:DRIVES|SIGNALS|IMPACTS]->
                            (t:Concept)
                RETURN path, c.id as company, t.id as trend,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """
            ]
            for query in company_queries:
                try:
                    paths = self.neo4j_graph.query(query, {"company": tags["company"]})
                    results.extend(paths)
                except Exception as e:
                    logger.error(f"[Neo4j][Company] Error in query: {str(e)}")

        # Sector-centric paths
        if "sector" in tags and tags["sector"]:
            sector_queries = [
                # Sector to companies
                """
                MATCH path = (s:Sector {id: $sector})
                            <-[:HAS_CONTEXT|HAS_SECTOR|OPERATES_IN]-
                            (c:Company)
                RETURN path, s.id as sector, c.id as company,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """,
                # Sector to signals
                """
                MATCH path = (s:Sector {id: $sector})
                            -[:RELATED_TO|DRIVES|IMPACTS]->
                            (sig:Concept)
                RETURN path, s.id as sector, sig.id as signal,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """,
                # Sector to trends
                """
                MATCH path = (s:Sector {id: $sector})
                            -[:DRIVES|IMPACTS]->
                            (t:Concept)
                RETURN path, s.id as sector, t.id as trend,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """
            ]
            for query in sector_queries:
                try:
                    paths = self.neo4j_graph.query(query, {"sector": tags["sector"]})
                    results.extend(paths)
                except Exception as e:
                    logger.error(f"[Neo4j][Sector] Error in query: {str(e)}")

        # Country-centric paths
        if "country" in tags and tags["country"]:
            try:
                query = """
                MATCH path = (co:Country {id: $country})
                            <-[:LOCATED_IN|BASED_IN]-
                            (c:Company)
                RETURN path, co.id as country, c.id as company
                LIMIT 3
                """
                paths = self.neo4j_graph.query(query, {"country": tags["country"]})
                results.extend(paths)
            except Exception as e:
                logger.error(f"[Neo4j][Country] Error in query: {str(e)}")

        # Documents related to any tags
        doc_conditions = []
        params = {}

        if tags.get("sector"):
            doc_conditions.append("(d)-[:CONTAINS]->(:Sector {id: $sector})")
            params["sector"] = tags["sector"]
        if tags.get("company"):
            doc_conditions.append("(d)-[:CONTAINS]->(:Company {id: $company})")
            params["company"] = tags["company"]
        if tags.get("country"):
            doc_conditions.append("(d)-[:CONTAINS]->(:Country {id: $country})")
            params["country"] = tags["country"]

        if doc_conditions:
            try:
                query = f"""
                MATCH (d:Document)
                WHERE {" OR ".join(doc_conditions)}
                RETURN d.id as id, d.title as title, d.source_url as url
                LIMIT 5
                """
                docs = self.neo4j_graph.query(query, params)
                results.extend(docs)
            except Exception as e:
                logger.error(f"[Neo4j][Document] Error in document query: {str(e)}")

        return results
        
    def trace_knowledge_paths(self, chunk_ids: list[str], depth: int = 2):
        """
        Traverse from Chunks and return enriched paths with full node details and relationship types,
        also extracting the text content from chunks for RAG context.
        """
        # First get the chunk texts directly
        chunk_text_query = """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids
        RETURN c.id AS chunk_id, c.text AS chunk_text
        """
        chunk_texts = self.neo4j_graph.query(chunk_text_query, {"chunk_ids": chunk_ids})
        
        # Log the chunk texts retrieved from Neo4j
        logger.info("=== NEO4J CHUNK TEXTS RETRIEVED ===")
        chunk_content = []
        for i, row in enumerate(chunk_texts):
            chunk_id = row.get("chunk_id", "unknown")
            text = row.get("chunk_text", "")
            # Clean HTML tags if present
            text = text.replace("<p>", "").replace("</p>", "\n").strip()
            if text:
                chunk_content.append({
                    "id": chunk_id,
                    "text": text,
                    "source": "neo4j_chunk"
                })
                logger.info(f"Neo4j Chunk {i+1}:")
                logger.info(f"  Chunk ID: {chunk_id}")
                logger.info(f"  Text preview: {text[:200]}...")
                logger.info("  " + "-" * 50)
        logger.info("=== END NEO4J CHUNK TEXTS ===")
        
        # Then get the knowledge graph paths
        cypher = f"""
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids
        CALL apoc.path.subgraphAll(c, {{
            maxLevel: {depth},
            relationshipFilter: '>, <',
            labelFilter: '+Company|+Country|+Concept|+Product|+Sector|+Subsector|+Document|+Consumer|+Location'
        }})
        YIELD nodes, relationships

        UNWIND relationships AS rel
        WITH 
            startNode(rel) AS start, 
            endNode(rel) AS end, 
            type(rel) AS rel_type

        RETURN 
            start.id AS start_id,
            labels(start)[0] AS start_type,
            coalesce(start.id) AS start_name,
            CASE 
                WHEN 'Chunk' IN labels(start) OR 'Document' IN labels(start) 
                THEN apoc.map.removeKey(properties(start), 'embedding')
                ELSE properties(start)
            END AS start_properties,

            rel_type AS relationship,

            end.id AS end_id,
            labels(end)[0] AS end_type,
            coalesce(end.id) AS end_name,
            CASE 
                WHEN 'Chunk' IN labels(end) OR 'Document' IN labels(end) 
                THEN apoc.map.removeKey(properties(end), 'embedding')
                ELSE properties(end)
            END AS end_properties
        """
        
        path_results = self.neo4j_graph.query(cypher, {"chunk_ids": chunk_ids})
        
        # Log the knowledge graph paths
        logger.info("=== NEO4J KNOWLEDGE GRAPH PATHS ===")
        for i, row in enumerate(path_results):
            logger.info(f"Path {i+1}:")
            logger.info(f"  {row.get('start_type', 'Unknown')} '{row.get('start_name', 'Unknown')}' " +
                       f"{row.get('relationship', 'related to').replace('_', ' ')} " +
                       f"{row.get('end_type', 'Unknown')} '{row.get('end_name', 'Unknown')}'")
            logger.info("  " + "-" * 50)
        logger.info("=== END NEO4J KNOWLEDGE GRAPH PATHS ===")
        
        return path_results, chunk_content

    def retrieve(self, refined_query: dict):
        """Retrieve from separate collections: regular posts + PDF (text/table) chunks + Neo4j chunks."""
        query_text = refined_query.get("refined_query", refined_query.get("original_query", ""))
        tags = refined_query.get("tags", {})
        self.current_tags = tags
        logger.info(f"[Retrieval] Query='{query_text}' Tags={tags}")

        # --- 1. Regular posts (tester2) ---
        post_hits = self.retrieve_from_qdrant(query_text, top_k=16)

        regular_chunks = []
        regular_docs_formatted = []
        for hit in post_hits:
            payload = hit.payload or {}
            if 'postId' not in payload:
                continue  # ensure only true posts
            chunk_id = "chunk_" + str(payload['postId'])
            regular_chunks.append(chunk_id)
            regular_docs_formatted.append({
                'id': payload['postId'],
                'text': payload.get('text', payload.get('content', '')),
                'score': hit.score,
                'source': 'regular_post',
                'metadata': payload
            })

        logger.info(f"[Retrieval] Regular posts processed: {len(regular_docs_formatted)}")

        # --- 2. PDF collection (veerive_docs) ---
        pdf_hits_raw = self.retrieve_from_pdf_docs(query_text, top_k=12)  # already filters & boosts
        pdf_docs_processed = self.process_pdf_results(pdf_hits_raw)
        logger.info(f"[Retrieval] PDF processed: {len(pdf_docs_processed)}")

        # --- 3. Graph reasoning (only meaningful if we have tags) ---
        reasoner_results = self.kg_reasoner.reason(tags) if tags else []
        logger.info(f"[Retrieval] KG insights: {len(reasoner_results)}")

        # --- 4. Direct knowledge graph paths from chunks + chunk text content ---
        pathscontext = []
        neo4j_chunks = []
        if regular_chunks:
            try:
                neo4j_paths, neo4j_chunks = self.trace_knowledge_paths(regular_chunks, depth=1)
                pathscontext = convert_paths_to_natural_language(neo4j_paths)
                logger.info(f"[Retrieval] Neo4j chunks with text: {len(neo4j_chunks)}")
            except Exception as e:
                logger.error(f"[KG paths] Error: {e}")

        # --- 5. Prompt guidance ---
        prompt_results = self.retrieve_prompt(query_text, top_k=1)

        # --- 6. Assemble result ---
        # Add Neo4j chunk content to the RAG context
        combined_docs = regular_docs_formatted + neo4j_chunks
        
        result = {
            "refined_query": refined_query,
            "qdrant_docs": combined_docs,  # Include both regular Qdrant and Neo4j chunks
            "pdf_docs": pdf_docs_processed,
            "pdf_content": self.format_pdf_content(pdf_docs_processed),
            "kg_insights": reasoner_results,
            "kg_paths": pathscontext,
            "prompt": prompt_results,
            "neo4j_chunks": neo4j_chunks,  # Include separately for completeness
        }

        # DEBUG DUMP (trim large fields)
        try:
            logger.info(
                "[Retrieval Summary] posts=%d neo4j=%d pdf=%d kg_insights=%d kg_paths=%d prompt=%d",
                len(regular_docs_formatted),
                len(neo4j_chunks),
                len(result["pdf_docs"]),
                len(result["kg_insights"]),
                len(result["kg_paths"]),
                len(result["prompt"]),
            )
        except Exception:
            pass

        return result
    
    def retrieve_from_pdf_docs(self, query_text: str, top_k: int = 8):
        """Retrieve relevant PDF document chunks with enhanced filtering and scoring"""
        try:
            query_vector = self.embedder.embed_query(query_text)
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.pdf_collection not in collection_names:
                logger.warning(f"PDF collection '{self.pdf_collection}' does not exist in Qdrant")
                return []
            
            # Use higher limit to get more candidates for filtering
            search_limit = min(top_k * 3, 20)
            
            hits = self.qdrant_client.search(
                collection_name=self.pdf_collection,
                query_vector=query_vector,
                limit=search_limit,
                with_payload=True,
                with_vectors=False,
                timeout=10,
                score_threshold=0.5,  # Lower threshold for PDFs as they might be more diverse
            )
            
            # Print detailed PDF chunk information
            logger.info("=== PDF CHUNKS RETRIEVED ===")
            for i, hit in enumerate(hits):
                payload = hit.payload or {}
                text_preview = payload.get('chunk_text', '')[:200]
                logger.info(f"PDF Chunk {i+1}:")
                logger.info(f"  Score: {hit.score:.4f}")
                logger.info(f"  Doc Title: {payload.get('doc_title', 'N/A')}")
                logger.info(f"  Is Table: {payload.get('is_table', False)}")
                logger.info(f"  Chunk ID: {payload.get('chunk_id', 'N/A')}")
                logger.info(f"  Text preview: {text_preview}...")
                logger.info(f"  Full payload keys: {list(payload.keys())}")
                logger.info("  " + "-" * 50)
            logger.info("=== END PDF CHUNKS ===")
            
            # Apply additional filtering for PDF documents
            filtered_hits = []
            current_tags = getattr(self, 'current_tags', {})
            
            for hit in hits:
                payload = hit.payload
                text_content = payload.get('chunk_text', '').lower()
                
                # Basic relevance check
                relevance_boost = 0
                
                # Check for tag matches
                if current_tags.get('sector'):
                    if current_tags['sector'].lower() in text_content:
                        relevance_boost += 0.1
                
                if current_tags.get('country'):
                    if current_tags['country'].lower() in text_content:
                        relevance_boost += 0.1
                
                # Boost tables for business model queries
                if payload.get('is_table') and current_tags.get('query_type', '').lower() == 'business models':
                    relevance_boost += 0.15
                
                # Apply boost to score
                adjusted_score = hit.score + relevance_boost
                hit.score = min(adjusted_score, 1.0)  # Cap at 1.0
                
                filtered_hits.append(hit)
            
            # Sort by adjusted score and return top_k
            filtered_hits.sort(key=lambda x: x.score, reverse=True)
            final_hits = filtered_hits[:top_k]
            
            logger.info(f"Found {len(final_hits)} PDF document results (from {len(hits)} candidates)")
            logger.info("=== FINAL FILTERED PDF CHUNKS ===")
            for i, hit in enumerate(final_hits):
                payload = hit.payload or {}
                logger.info(f"Final PDF Chunk {i+1}: Score={hit.score:.4f}, Title={payload.get('doc_title', 'N/A')}")
            logger.info("=== END FINAL PDF CHUNKS ===")
            
            return final_hits
            
        except Exception as e:
            logger.error(f"Error retrieving from PDF documents: {str(e)}")
            return []

    def format_pdf_content(self, pdf_chunks: list) -> str:
        """Format PDF chunks into specialized content sections for the generator"""
        if not pdf_chunks:
            return ""
        
        formatted_content = "\n\n=== SPECIALIZED PDF DOCUMENT INSIGHTS ===\n\n"
        
        # Separate tables and text content
        tables = [doc for doc in pdf_chunks if doc['is_table']]
        text_docs = [doc for doc in pdf_chunks if not doc['is_table']]
        
        # Group by document title
        docs_by_title = {}
        for doc in pdf_chunks:
            title = doc['title']
            if title not in docs_by_title:
                docs_by_title[title] = {
                    'title': title,
                    'url': doc['source_url'],
                    'tables': [],
                    'text_chunks': [],
                    'max_score': 0
                }
            
            docs_by_title[title]['max_score'] = max(docs_by_title[title]['max_score'], doc['score'])
            
            if doc['is_table']:
                docs_by_title[title]['tables'].append(doc)
            else:
                docs_by_title[title]['text_chunks'].append(doc)
        
        # Sort documents by relevance
        sorted_docs = sorted(docs_by_title.values(), key=lambda x: x['max_score'], reverse=True)
        
        # Format each document section
        for i, doc_info in enumerate(sorted_docs[0:]):  # Top 3 most relevant documents
            formatted_content += f"PDF Document {i+1}: {doc_info['title']}\n"
            formatted_content += f"Source: {doc_info['url']}\n"
            formatted_content += f"Relevance Score: {doc_info['max_score']:.3f}\n\n"
            
            # Add table data first (often most structured)
            if doc_info['tables']:
                formatted_content += "📊 TABLE DATA:\n"
                for table in doc_info['tables'][0:]:  # Max 2 tables per document
                    formatted_content += f"{table['formatted_content']}\n"
                formatted_content += "\n"
            
            # Add text content
            if doc_info['text_chunks']:
                formatted_content += "📄 TEXT CONTENT:\n"
                for chunk in doc_info['text_chunks'][0:]:  # Max 2 text chunks per document
                    formatted_content += f"{chunk['formatted_content'][:800]}...\n\n"
            
            formatted_content += "=" * 50 + "\n\n"
        
        # Add summary section for high-level insights
        if len(pdf_chunks) > 0:
            high_score_docs = [doc for doc in pdf_chunks if doc['score'] > 0.75]
            if high_score_docs:
                formatted_content += "🔍 HIGH CONFIDENCE PDF INSIGHTS:\n"
                for doc in high_score_docs[0:]:
                    formatted_content += f"• {doc['title']}: {doc['formatted_content'][:200]}...\n"
                formatted_content += "\n"
        
        return formatted_content
    
    def process_pdf_results(self, pdf_results: list) -> list:
        """Process PDF results with enhanced categorization and formatting"""
        processed_pdfs = []
        print(f"Processing {len(pdf_results)} PDF results with enhanced formatting")
        print(pdf_results)

        for result in pdf_results:
            payload = result.payload
            
            # Extract and categorize PDF content
            pdf_doc = {
                'id': payload.get('doc_id', str(result.id)),
                'title': payload.get('doc_title', 'Unknown PDF'),
                'text': payload.get('chunk_text', ''),
                'score': result.score,
                'is_table': payload.get('is_table', False),
                'source_url': payload.get('source_url', ''),
                'post_type': payload.get('post_type', ''),
                'sentiment': payload.get('sentiment', ''),
                'chunk_id': payload.get('chunk_id', 0),
                'source': 'pdf_document'
            }
            
            # Add content type classification
            if pdf_doc['is_table']:
                pdf_doc['content_type'] = 'table'
                pdf_doc['formatted_content'] = self.format_table_content(pdf_doc['text'])
            else:
                pdf_doc['content_type'] = 'text'
                pdf_doc['formatted_content'] = self.format_text_content(pdf_doc['text'])
            
            # Add relevance indicators
            pdf_doc['relevance_score'] = self.calculate_pdf_relevance(pdf_doc, self.current_tags)
            
            processed_pdfs.append(pdf_doc)
        
        # Sort by combined score (original score + relevance)
        processed_pdfs.sort(
            key=lambda x: (x['score'] * 0.7 + x['relevance_score'] * 0.3), 
            reverse=True
        )
        
        logger.info(f"Processed {len(processed_pdfs)} PDF documents with enhanced formatting")
        return processed_pdfs
    
    def format_table_content(self, table_text: str) -> str:
        """Format table content for better readability"""
        if not table_text:
            return ""
        
        # Add table formatting indicators
        formatted = f"[TABLE DATA]\n{table_text}\n[/TABLE DATA]"
        return formatted
    
    def format_text_content(self, text: str) -> str:
        """Format text content with better structure"""
        if not text:
            return ""
        
        # Clean and structure the text
        cleaned_text = text.strip()
        
        # Add paragraph breaks for long text
        if len(cleaned_text) > 500:
            # Try to break at sentence endings
            sentences = cleaned_text.split('. ')
            if len(sentences) > 3:
                mid_point = len(sentences) // 2
                cleaned_text = '. '.join(sentences[:mid_point]) + '.\n\n' + '. '.join(sentences[mid_point:])
        
        return cleaned_text
    
    def calculate_pdf_relevance(self, pdf_doc: dict, tags: dict) -> float:
        """Calculate additional relevance score for PDF documents based on tags"""
        relevance_score = 0.0
        
        text_content = pdf_doc['text'].lower()
        title_content = pdf_doc['title'].lower()
        
        # Check for tag matches in content
        if tags.get('sector'):
            sector = tags['sector'].lower()
            if sector in text_content:
                relevance_score += 0.3
            if sector in title_content:
                relevance_score += 0.2
        
        if tags.get('country'):
            country = tags['country'].lower()
            if country in text_content:
                relevance_score += 0.2
            if country in title_content:
                relevance_score += 0.15
        
        if tags.get('company'):
            company = tags['company'].lower()
            if company in text_content:
                relevance_score += 0.25
            if company in title_content:
                relevance_score += 0.2
        
        if tags.get('subsector'):
            subsector = tags['subsector'].lower()
            if subsector in text_content:
                relevance_score += 0.15
        
        # Bonus for tables when looking for business models or financial data
        query_type = tags.get('query_type', '').lower()
        if pdf_doc['is_table'] and ('business model' in query_type or 'financial' in query_type):
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)  # Cap at 1.0