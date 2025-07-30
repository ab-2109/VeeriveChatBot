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

    def search_prompt(self, query_text: str, sector: str = None, subsector: str = None, top_k: int = 5):
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
        query = f"""
        MATCH path = (c:Company {{id: "{company}"}})
                    -[:SUPPORTS|SERVES|DRIVES|IMPACTS*1..2]-(e)
        RETURN path, c.id as company, labels(e)[0] as entity_type,
               e.id as related_entity, type(relationships(path)[-1]) as relationship_type
        LIMIT 10
        """
        return self.graph.query(query)

    def reason_over_sector_trends(self, sector: str):
        query = f"""
        MATCH path = (s:Contextsector {{id: "{sector}"}})
                    -[:SIGNALS|DRIVES|IMPACTS*1..2]-(sig:Signal)
                    -[:PROVIDES_NEWS|MENTIONS*0..1]-(n:News)
        RETURN path, s.id as sector, sig.id as signal,
               
               CASE WHEN n IS NOT NULL THEN n.id ELSE NULL END as news_id
        LIMIT 5
        """
        return self.graph.query(query)

    def reason_market_trends(self, signal: str):
        query = f"""
        MATCH path = (sig:Signal {{id: "{signal}"}})
                    -[:IMPACTS|DRIVES*1..2]-(e)
        RETURN path, sig.id as signal,
               labels(e)[0] as impacted_type,
               e.id as impacted_entity,
               type(relationships(path)[-1]) as relationship
        LIMIT 8
        """
        return self.graph.query(query)

    def reason_over_product_impact(self, product: str):
        query = f"""
        MATCH path = (p:Product {{id: "{product}"}})
                    -[:IMPACTS|DRIVES|SUPPORTS|SERVES*1..2]-(e)
        RETURN path, p.id as product,
               labels(e)[0] as impacted_type,
               e.id as impacted_entity,
               type(relationships(path)[-1]) as relationship_type
        LIMIT 8
        """
        return self.graph.query(query)

    def reason_over_business_model(self, sector: str, country: Optional[str] = None):
        country_clause = f"""
        MATCH (company:Company)-[:LOCATED_IN]->(country:Country {{id: "{country}"}})
        """ if country else ""

        query = f"""
        MATCH (s:Contextsector {{id: "{sector}"}})
        MATCH (company:Company)-[:HAS_CONTEXT|HAS_SECTOR]->(s)
        {country_clause}
        OPTIONAL MATCH (company)-[:SUPPORTS|SERVES]->(consumer:Consumer)
        OPTIONAL MATCH (company)-[r:DRIVES|IMPACTS]->(trend:Trend)
        RETURN company.id as company,
               COLLECT(DISTINCT consumer.id) as consumers,
               COLLECT(DISTINCT {{trend: trend.id, rel: type(r), desc: r.description}}) as trends
        LIMIT 5
        """
        return self.graph.query(query)

    def reason_document_insights(self, tags: Dict[str, str]):
        conditions = []
        params = {}

        if tags.get("company"):
            conditions.append("(d)-[:CONTAINS]->(:Company {id: $company})")
            params["company"] = tags["company"]

        if tags.get("sector"):
            conditions.append("(d)-[:CONTAINS|HAS_CONTEXT]->(:Contextsector {id: $sector})")
            params["sector"] = tags["sector"]

        if tags.get("country"):
            conditions.append("(d)-[:CONTAINS]->(:Country {id: $country})")
            params["country"] = tags["country"]

        if not conditions:
            return self.graph.query("MATCH (d:Document) RETURN d.id as id, d.title as title, d.summary as summary, d.source_url as url LIMIT 5")

        query = f"""
        MATCH (d:Document)
        WHERE {" AND ".join(conditions)}
        RETURN d.id as id, d.title as title, d.summary as summary, d.source_url as url
        LIMIT 5
        """
        return self.graph.query(query, params)

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
                 pdf_collection: str = "veerive_docs",  # Add PDF collection
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
        self.pdf_collection = pdf_collection  # Store PDF collection name

        self.embedder = OpenAIEmbeddings(model=embed_model, api_key=os.getenv("OPENAI_API_KEY"))

        # Neo4j
        self.neo4j_graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pass)
        
        # Initialize the knowledge graph reasoner
        self.kg_reasoner = KGReasoner(self.neo4j_graph)


    def retrieve_from_qdrant(self, query_text: str, top_k: int = 5):
        """Retrieve from regular posts collection with enhanced scoring"""
        try:
            query_vector = self.embedder.embed_query(query_text)
            hits = self.qdrant_client.search(
                collection_name=self.qdrant_collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                timeout=10,
                score_threshold=0.55,
            )
            
            logger.info(f"Found {len(hits)} regular document results")
            return hits
        except Exception as e:
            logger.error(f"Error retrieving from regular collection: {str(e)}")
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
                            -[:SUPPORTS|SERVES|DRIVES|CONTAINS]->
                            (p:Product)
                RETURN path, c.id as company, p.id as product,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """,
                # Company to sectors
                """
                MATCH path = (c:Company {id: $company})
                            -[:HAS_CONTEXT|HAS_SECTOR]->
                            (s:Contextsector)
                RETURN path, c.id as company, s.id as sector,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """,
                # Company to trends
                """
                MATCH path = (c:Company {id: $company})
                            -[:DRIVES|SIGNALS|IMPACTS]->
                            (t:Trend)
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
                MATCH path = (s:Contextsector {id: $sector})
                            <-[:HAS_CONTEXT|HAS_SECTOR|OPERATES_IN]-
                            (c:Company)
                RETURN path, s.id as sector, c.id as company,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """,
                # Sector to signals
                """
                MATCH path = (s:Contextsector {id: $sector})
                            -[:SIGNALS|DRIVES|IMPACTS]->
                            (sig:Signal)
                RETURN path, s.id as sector, sig.id as signal,
                    type(relationships(path)[0]) as relationship
                LIMIT 3
                """,
                # Sector to trends
                """
                MATCH path = (s:Contextsector {id: $sector})
                            -[:DRIVES|IMPACTS]->
                            (t:Trend)
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
            doc_conditions.append("(d)-[:CONTAINS]->(:Contextsector {id: $sector})")
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
        excluding the 'embedding' property from Chunk and Document nodes.
        """
        cypher = f"""
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids
        CALL apoc.path.subgraphAll(c, {{
            maxLevel: {depth},
            relationshipFilter: '>, <',
            labelFilter: '+Company|+Country|+Signal|+Trend|+Product|+Contextsector|+Subsector|+Document'
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
        return self.neo4j_graph.query(cypher, {"chunk_ids": chunk_ids})

        
    def convert_paths_to_natural_language(path_rows):
        """
        Convert raw Cypher query results into readable natural language sentences.
        """
        statements = []

        for row in path_rows:
            start = row.get("start_name", "Unknown")
            start_type = row.get("start_type", "")
            end = row.get("end_name", "Unknown")
            end_type = row.get("end_type", "")
            rel = row.get("relationship", "related to")

            # Get optional properties
            start_props = row.get("start_properties", {})
            end_props = row.get("end_properties", {})

            # Begin with relationship summary
            sentence = f"{start_type} '{start}' {rel.replace('_', ' ').lower()} {end_type} '{end}'."

            # Enrich with extra info
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



    def retrieve(self, refined_query: dict):
        """Retrieve relevant information using all available data sources with separate handling for PDFs"""
        query_text = refined_query.get("refined_query", refined_query.get("original_query", ""))
        tags = refined_query.get("tags", {})
        print(f"Tags: {tags}")
        query_type = tags.get("query_type", "")
        
        # Store the current tags for use in retrieve_prompt
        self.current_tags = tags
        
        # Retrieve from regular posts and PDF documents separately
        regular_results = self.retrieve_from_qdrant(query_text, top_k=8)
        pdf_results = self.retrieve_from_pdf_docs(query_text, top_k=5)
        
        # Process regular chunks for graph reasoning
        regular_chunks = []
        regular_docs_formatted = []
        
        for result in regular_results:
            chunk_id = "chunk_" + str(result.payload['postId'])
            regular_chunks.append(chunk_id)
            regular_docs_formatted.append({
                'id': result.payload['postId'],
                'text': result.payload.get('text', result.payload.get('content', '')),
                'score': result.score,
                'source': 'regular_post',
                'metadata': result.payload
            })
        
        # Process PDF chunks with enhanced formatting and categorization
        pdf_docs_processed = self.process_pdf_results(pdf_results)
        
        # Get graph insights using the reasoner (only for regular posts that have graph connections)
        reasoner_results = self.kg_reasoner.reason(tags)
        
        # Get direct graph paths (only for regular chunks that have graph relationships)
        neo4j_paths = []
        pathscontext = []
        if regular_chunks:
            neo4j_paths = self.trace_knowledge_paths(regular_chunks, 1)
            pathscontext = convert_paths_to_natural_language(neo4j_paths)

        # Retrieve prompt guidance with error handling
        prompt_results = self.retrieve_prompt(query_text, 1)
        
        return {
            "refined_query": refined_query,
            "qdrant_docs": regular_docs_formatted,  # Only regular posts for normal processing
            "pdf_docs": pdf_docs_processed,  # Specially processed PDF documents
            "pdf_content": self.format_pdf_content(pdf_docs_processed),  # Formatted PDF content
            "kg_insights": reasoner_results,
            "kg_paths": pathscontext,
            "prompt": prompt_results,
        }
    
    def retrieve_from_pdf_docs(self, query_text: str, top_k: int = 5):
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
        for i, doc_info in enumerate(sorted_docs[:3]):  # Top 3 most relevant documents
            formatted_content += f"PDF Document {i+1}: {doc_info['title']}\n"
            formatted_content += f"Source: {doc_info['url']}\n"
            formatted_content += f"Relevance Score: {doc_info['max_score']:.3f}\n\n"
            
            # Add table data first (often most structured)
            if doc_info['tables']:
                formatted_content += "📊 TABLE DATA:\n"
                for table in doc_info['tables'][:2]:  # Max 2 tables per document
                    formatted_content += f"{table['formatted_content']}\n"
                formatted_content += "\n"
            
            # Add text content
            if doc_info['text_chunks']:
                formatted_content += "📄 TEXT CONTENT:\n"
                for chunk in doc_info['text_chunks'][:2]:  # Max 2 text chunks per document
                    formatted_content += f"{chunk['formatted_content'][:800]}...\n\n"
            
            formatted_content += "=" * 50 + "\n\n"
        
        # Add summary section for high-level insights
        if len(pdf_chunks) > 0:
            high_score_docs = [doc for doc in pdf_chunks if doc['score'] > 0.75]
            if high_score_docs:
                formatted_content += "🔍 HIGH CONFIDENCE PDF INSIGHTS:\n"
                for doc in high_score_docs[:3]:
                    formatted_content += f"• {doc['title']}: {doc['formatted_content'][:200]}...\n"
                formatted_content += "\n"
        
        return formatted_content
    
    def process_pdf_results(self, pdf_results: list) -> list:
        """Process PDF results with enhanced categorization and formatting"""
        processed_pdfs = []
        
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
def convert_paths_to_natural_language(path_rows):
    """
    Convert raw Cypher query results into readable natural language sentences.
    """
    statements = []

    for row in path_rows:
        start = row.get("start_name", "Unknown")
        start_type = row.get("start_type", "")
        end = row.get("end_name", "Unknown")
        end_type = row.get("end_type", "")
        rel = row.get("relationship", "related to")

        # Get optional properties
        start_props = row.get("start_properties", {})
        end_props = row.get("end_properties", {})

        # Begin with relationship summary
        sentence = f"{start_type} '{start}' {rel.replace('_', ' ').lower()} {end_type} '{end}'."

        # Enrich with extra info
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


if __name__ == "__main__":
    # Example usage
    username = "chaubeyp"
    password = urllib.parse.quote_plus("ConsTrack360")
    mongouri = f"mongodb+srv://{username}:{password}@veerive.tta8g.mongodb.net/"
    mongo_uri = mongouri
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_pass = os.getenv("NEO4J_PASSWORD")

    retrieval_agent = RetrievalAgent(mongo_uri, qdrant_url, qdrant_key, neo4j_uri, neo4j_user, neo4j_pass)
    
    # Example refined query
    refined_query = {
      "original_query": "What are the dominant business models in B2C BNPL in India?",
      "refined_query": "What are the dominant business models for Buy Now, Pay Later (BNPL) companies serving B2C customers in India?", 
      "tags": {
        "sector": "BNPL",
        "country": "India",
        "company": "",
        "subsector": "B2C",
        "query_type": "Business Models"
      }
    }

    results = retrieval_agent.retrieve(refined_query)
    print(f"Found {len(results['qdrant_docs'])} regular document results")
    print(f"Found {len(results['pdf_docs'])} PDF document results")
    print(f"Found {len(results['kg_insights'])} knowledge graph insights")
    print(f"Found {len(results['kg_paths'])} direct graph paths")
    
    # Show regular documents
    print("\n=== REGULAR DOCUMENTS ===")
    for doc in results['qdrant_docs']:
        print(f"ID: {doc['id']}, Score: {doc['score']:.3f}, Source: {doc['source']}")
    
    # Show PDF documents with enhanced info
    print("\n=== PDF DOCUMENTS ===")
    for doc in results['pdf_docs']:
        print(f"Title: {doc['title']}, Score: {doc['score']:.3f}, Type: {doc['content_type']}, Relevance: {doc['relevance_score']:.3f}")
    
    # Show formatted PDF content
    if results['pdf_content']:
        print("\n=== FORMATTED PDF CONTENT (first 500 chars) ===")
        print(results['pdf_content'][:500] + "...")
