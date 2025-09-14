from langgraph.graph import StateGraph
from typing import TypedDict, Dict, Any, List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime
from agents.intake import process_intake
from agents.refiner import get_refiner
from agents.retrieval import RetrievalAgent
from agents.raggen import run_rag_generator
from agents.clarification import clarification_node, process_clarification_answer
import operator
import logging
import urllib.parse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('graph_processor')

load_dotenv()

class GraphState(TypedDict):
    query: str  
    metadata: Dict[str, Any] 
    intake_state: Optional[Dict[str, Any]]
    refined_query: Optional[Dict[str, Any]]
    retrieval_results: Optional[Dict[str, Any]]
    generated_response: Optional[Dict[str, Any]]
    clarification_question: Optional[str]
    clarification_answers: Optional[List[Dict[str, str]]] 
    status: Optional[str]
    errors: Optional[List[str]] 

def init_retrieval_agent():
    try:
        username = os.getenv("MONGO_USERNAME")
        password = urllib.parse.quote_plus(os.getenv("MONGO_PASSWORD"))
        mongo_uri = f"mongodb+srv://{username}:{password}@veerive.tta8g.mongodb.net/"
        
        return RetrievalAgent(
            mongo_uri=mongo_uri,
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_key=os.getenv("QDRANT_API"),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USERNAME"),
            neo4j_pass=os.getenv("NEO4J_PASSWORD"),
            qdrant_collection="tester2", 
            pdf_collection="veerive_docs",  
            embed_model="text-embedding-3-large"
        )
    except Exception as e:
        logger.error(f"Failed to initialize retrieval agent: {str(e)}")
        raise

def intake_node(state: GraphState) -> GraphState:
    try:
        query = state["query"]
        metadata = state["metadata"]
        
        logger.info(f"Processing intake for query: {query[:100]}...")

        result = process_intake({
            "query": query,
            "metadata": metadata
        }, interactive=False)

        if result["status"] == "error":
            logger.error(f"Intake error: {result['message']}")
            return {
                "errors": [f"Intake error: {result['message']}"],
                "status": "error"
            }

        logger.info("Intake processing completed successfully")
        return {
            "intake_state": result["data"],
            "status": "intake_complete"
        }
    except Exception as e:
        logger.error(f"Intake processing failed: {str(e)}")
        return {
            "errors": [f"Intake processing failed: {str(e)}"],
            "status": "error"
        }

def clarification_node_wrapper(state: GraphState) -> GraphState:
    try:
        query = state.get("query", "")
        
        metadata = state.get("metadata", {})
        
        intake_state = state.get("intake_state")
        if intake_state:
            metadata["intake_state"] = intake_state
        
        clarification_state = {
            "query": query,
            "metadata": metadata
        }
        
        result = clarification_node(clarification_state)
        
        updated_state = {**state}
        
        if "clarification_question" in result:
            updated_state["clarification_question"] = result["clarification_question"]
        
        if "status" in result and result["status"]:
            updated_state["status"] = result["status"]
        
        return updated_state
    except Exception as e:
        return {**state, "errors": state.get("errors", []) + [f"Clarification error: {str(e)}"]}

def refine_node(state: GraphState) -> GraphState:
    try:
        metadata = state.get("metadata", {})
        
        logger.info("Starting query refinement")
        
        if "clarifications" in metadata and metadata["clarifications"]:
            if state.get("intake_state"):
                intake_state = state.get("intake_state", {})
                intake_state["metadata"] = metadata
            else:
                original_query = metadata.get("original_query", state["query"])
                intake_state = {
                    "query": original_query,
                    "metadata": metadata
                }
        else:
            intake_state = state.get("intake_state", {})
        
        refined = get_refiner().refine(intake_state)
        
        if "error" in refined:
            logger.error(f"Refinement error: {refined['error']}")
            return {
                "errors": [refined["error"]],
                "status": "error"
            }
        
        logger.info("Query refinement completed successfully")
        return {
            "refined_query": refined,
            "status": "refine_complete"
        }
    except Exception as e:
        logger.error(f"Refine error: {str(e)}")
        return {
            "errors": [f"Refine error: {str(e)}"],
            "status": "error"
        }

def retrieve_node(state: GraphState) -> GraphState:
    try:
        refined_query = state.get("refined_query", {})
        logger.info(f"Starting retrieval for refined query: {refined_query.get('refined_query', '')[:100]}...")

        retrieval_agent = init_retrieval_agent()


        tags = refined_query.get("tags", {}) if isinstance(refined_query, dict) else {}
        tags_lower = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in (tags or {}).items()}

        if tags_lower.get("country") and not retrieval_agent.country_present(tags_lower):
            logger.info("Country not present/connected in Neo4j; short-circuit to friendly message.")
            empty_results = {
                "qdrant_docs": [],
                "pdf_docs": [],
                "kg_insights": [],
                "kg_paths": [],
                "prompt": [],
                "country_gate_message": "No information available regarding this country in the database.",
            }
            return {
                "retrieval_results": empty_results,
                "country_gate_message": "No information available regarding this country in the database.",
                "status": "retrieval_complete"
            }

        results = retrieval_agent.retrieve(refined_query)
        return {"retrieval_results": results, "status": "retrieval_complete"}
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        return {"errors": [f"Retrieval error: {str(e)}"], "status": "error"}

def generate_node(state: GraphState) -> GraphState:
    """Generate final response using enhanced retrieval results including PDF documents"""
    try:
        retrieval_results = state.get("retrieval_results", {}) or {}
        refined_query = state.get("refined_query", {"refined_query": state["query"]})

        gate_msg = state.get("country_gate_message") or retrieval_results.get("country_gate_message")
        if gate_msg:
            logger.info("Country gate message detected; short-circuiting generation with friendly message.")
            friendly = {
                "conversational": gate_msg,
                "structured": {"data": {}},
                "pdf": {"tables": [], "texts": [], "combined_markdown": ""}
            }
            return {"generated_response": friendly, "status": "complete"}

        generator_input = {
            "refined_query": refined_query,
            "qdrant_docs": retrieval_results.get("qdrant_docs", []),
            "pdf_docs": retrieval_results.get("pdf_docs", []),
            "kg_paths": retrieval_results.get("kg_paths", []),
            "kg_insights": retrieval_results.get("kg_insights", []),
            "prompt": retrieval_results.get("prompt", [])
        }
        if gate_msg:
            generator_input["country_gate_message"] = gate_msg

        result = run_rag_generator(generator_input)
        logger.info("Response generation completed successfully")
        if isinstance(result, dict) and "structured" not in result and "conversational" not in result:
            result = {"structured": {"data": {}}, "conversational": str(result)}
        return {"generated_response": result, "status": "complete"}
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return {"errors": [f"Generation error: {str(e)}"], "status": "error"}

def should_continue(state: GraphState) -> str:
    """Determine the next step in the graph based on current state"""
    if state.get("errors"):
        return "error"
        
    if "clarification_question" in state and state["clarification_question"]:
        return "needs_clarification"
    
    if state.get("intake_state") and not state.get("refined_query"):
        return "continue"  
    elif state.get("refined_query") and not state.get("retrieval_results"):
        return "continue_to_retrieve" 
    elif state.get("retrieval_results") and not state.get("generated_response"):
        return "continue_to_generate" 
    elif state.get("generated_response"):
        return "end" 
    
    return "continue"  

def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)
    
    # Add all nodes
    builder.add_node("intake", intake_node)
    builder.add_node("clarification", clarification_node_wrapper)
    builder.add_node("refine", refine_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    
    # Set entry point
    builder.set_entry_point("intake")
    
    builder.add_conditional_edges(
        "intake",
        should_continue,
        {
            "error": "__end__",
            "continue": "clarification",
            # Add a fallback for unexpected values
            "_fallback": "clarification"
        }
    )
    
    builder.add_conditional_edges(
        "clarification",
        should_continue,
        {
            "error": "__end__",
            "needs_clarification": "__end__",
            "continue": "refine", 
            "_fallback": "refine"
        }
    )
    
    builder.add_conditional_edges(
        "refine",
        should_continue,
        {
            "error": "__end__",
            "continue_to_retrieve": "retrieve"
        }
    )
    
    builder.add_conditional_edges(
        "retrieve",
        should_continue,
        {
            "error": "__end__",
            "continue_to_generate": "generate"
        }
    )
    
    builder.add_conditional_edges(
        "generate",
        should_continue,
        {
            "error": "__end__",
            "end": "__end__"
        }
    )
    
    return builder.compile()

def process_query(query: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a new user query with enhanced logging and error handling"""
    try:
        logger.info(f"Starting query processing: {query[:100]}...")
        
        # Initialize the graph
        graph = build_graph()
        
        # Run the graph with the query
        result = graph.invoke({
            "query": query,
            "metadata": metadata or {},
            "errors": []
        })
        
        logger.info("Query processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        return {
            "status": "error", 
            "errors": [f"Failed to process query: {str(e)}"]
        }

def process_clarification(session_id: str, clarification_answer: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a clarification answer and continue the graph execution"""
    try:
        logger.info(f"Processing clarification answer for session {session_id}")
        
        original_query = metadata.get("original_query", "")
        
        updated_metadata = process_clarification_answer(original_query, clarification_answer, metadata or {})
        
        if "original_query" not in updated_metadata and original_query:
            updated_metadata["original_query"] = original_query
            
        graph = build_graph()
        result = graph.invoke({
            "query": original_query,  
            "metadata": updated_metadata,
            "errors": []
        })
        
        logger.info("Clarification processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to process clarification: {str(e)}")
        return {
            "status": "error", 
            "errors": [f"Failed to process clarification: {str(e)}"]
        }

