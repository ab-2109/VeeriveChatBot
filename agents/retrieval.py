from qdrant_client import QdrantClient
from pymongo import MongoClient
from langchain_community.graphs import Neo4jGraph
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import os
import urllib.parse
import logging
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retrieval_agent')
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_openai import OpenAIEmbeddings
import os

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

        self.embedder = OpenAIEmbeddings(model=embed_model, api_key=os.getenv("OPENAI_API_KEY"))

        # Neo4j
        self.neo4j_graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pass)
        
        # Initialize the knowledge graph reasoner
        self.kg_reasoner = KGReasoner(self.neo4j_graph)


    def retrieve_from_qdrant(self, query_text: str, top_k: int = 5):
        query_vector = self.embedder.embed_query(query_text)
        hits = self.qdrant_client.search(
            collection_name=self.qdrant_collection,
            query_vector=query_vector,
            limit=5,
            with_payload=True,
            with_vectors=False,
            timeout=10,
            score_threshold=0.55,
        )
        return hits
    
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
        """Retrieve relevant information using all available data sources"""
        query_text = refined_query.get("refined_query", refined_query.get("original_query", ""))
        tags = refined_query.get("tags", {})
        print(f"Tags: {tags}")
        query_type = tags.get("query_type", "")
        
        # Store the current tags for use in retrieve_prompt
        self.current_tags = tags
        
        # Get results from Qdrant vector search
        qdrant_results = self.retrieve_from_qdrant(query_text)
        chunks = []
        for result in qdrant_results:
            chunks.append("chunk_" + str(result.payload['postId']))
            
        # Get graph insights using the reasoner
        reasoner_results = self.kg_reasoner.reason(tags)
        
        # Get direct graph paths
        neo4j_paths = self.trace_knowledge_paths(chunks, 1)
        pathscontext = convert_paths_to_natural_language(neo4j_paths)

        # Retrieve prompt guidance with error handling
        prompt_results = self.retrieve_prompt(query_text, 1)
        
        return {
            "refined_query": refined_query,
            "qdrant_docs": qdrant_results,
            "kg_insights": reasoner_results,
            "kg_paths": pathscontext,
            "prompt": prompt_results,
        }
    
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
    print(f"Found {len(results['qdrant_docs'])} vector search results")
    print(f"Found {len(results['kg_insights'])} knowledge graph insights")
    print(f"Found {len(results['kg_paths'])} direct graph paths")
    print(f"Results: {results}")
    for doc in results['qdrant_docs']:
        print("ID:", doc.payload['postId'], "score:", doc.score)
