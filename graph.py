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
    query: str  # Single value - no annotation needed
    metadata: Dict[str, Any]  # Regular dictionary type without annotation
    intake_state: Optional[Dict[str, Any]]
    refined_query: Optional[Dict[str, Any]]
    retrieval_results: Optional[Dict[str, Any]]
    generated_response: Optional[Dict[str, Any]]
    clarification_question: Optional[str]
    clarification_answers: Optional[List[Dict[str, str]]]  # No annotation
    status: Optional[str]
    errors: Optional[List[str]]  # No annotation

def init_retrieval_agent():
    """Initialize RetrievalAgent with enhanced PDF support and better error handling"""
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
            qdrant_collection="tester2",  # Regular posts collection
            pdf_collection="veerive_docs",  # PDF documents collection
            embed_model="text-embedding-3-large"
        )
    except Exception as e:
        logger.error(f"Failed to initialize retrieval agent: {str(e)}")
        raise

def intake_node(state: GraphState) -> GraphState:
    """Process initial query intake"""
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
    """Wrapper around clarification node to ensure proper state handling"""
    try:
        # Get the query from state
        query = state.get("query", "")
        
        # Get metadata
        metadata = state.get("metadata", {})
        
        # Get intake state - add to metadata if needed
        intake_state = state.get("intake_state")
        if intake_state:
            metadata["intake_state"] = intake_state
        
        # Create a single state object for the clarification node
        clarification_state = {
            "query": query,
            "metadata": metadata
        }
        
        # Call clarification node with the correct arguments
        result = clarification_node(clarification_state)
        
        # Properly update state with result
        updated_state = {**state}
        
        # Add clarification question if present
        if "clarification_question" in result:
            updated_state["clarification_question"] = result["clarification_question"]
        
        # Make sure we're not setting status to None
        if "status" in result and result["status"]:
            updated_state["status"] = result["status"]
        
        return updated_state
    except Exception as e:
        # Handle any exceptions
        return {**state, "errors": state.get("errors", []) + [f"Clarification error: {str(e)}"]}

def refine_node(state: GraphState) -> GraphState:
    """Refine the query based on intake and clarifications"""
    try:
        metadata = state.get("metadata", {})
        logger.info("Starting query refinement")
        
        # If we've already got clarifications, use them with the original query
        if "clarifications" in metadata and metadata["clarifications"]:
            # Either use intake_state or build minimal state with original query
            if state.get("intake_state"):
                intake_state = state.get("intake_state", {})
                intake_state["metadata"] = metadata
            else:
                # Create a minimal intake state using the original query
                original_query = metadata.get("original_query", state["query"])
                intake_state = {
                    "query": original_query,
                    "metadata": metadata
                }
        else:
            # Normal flow from intake
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
    except FileNotFoundError as fe:
        logger.error(
            "Refine file missing: %s (cwd=%s)", getattr(fe, "filename", "<unknown>"), os.getcwd(),
            exc_info=True
        )
        return {
            "errors": [f"Refine missing file: {getattr(fe, 'filename', '<unknown>')} (cwd={os.getcwd()})"],
            "status": "error"
        }
    except Exception as e:
        logger.exception("Refine failed")
        return {
            "errors": [f"Refine errorososososososos: {type(e).__name__}: {str(e)}"],
            "status": "error"
        }

def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant information using enhanced RetrievalAgent with PDF support"""
    try:
        refined_query = state.get("refined_query", {})
        
        logger.info(f"Starting retrieval for refined query: {refined_query.get('refined_query', '')[:100]}...")
        
        retrieval_agent = init_retrieval_agent()
        results = retrieval_agent.retrieve(refined_query)
        
        # Log retrieval statistics
        qdrant_count = len(results.get("qdrant_docs", []))
        pdf_count = len(results.get("pdf_docs", []))
        kg_insights_count = len(results.get("kg_insights", []))
        kg_paths_count = len(results.get("kg_paths", []))
        prompt_count = len(results.get("prompt", []))
        
        logger.info(f"Retrieval completed: {qdrant_count} vector docs, {pdf_count} PDF docs, "
                   f"{kg_insights_count} KG insights, {kg_paths_count} KG paths, {prompt_count} prompts")
        
        return {
            "retrieval_results": results,
            "status": "retrieval_complete"
        }
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        return {
            "errors": [f"Retrieval error: {str(e)}"],
            "status": "error"
        }

def generate_node(state: GraphState) -> GraphState:
    """Generate final response using enhanced retrieval results including PDF documents"""
    try:
        retrieval_results = state.get("retrieval_results", {})
        refined_query = state.get("refined_query", {"refined_query": state["query"]})
        
        # Prepare input for the generator with enhanced data sources
        generator_input = {
            "refined_query": refined_query,
            "qdrant_docs": retrieval_results.get("qdrant_docs", []),
            "pdf_docs": retrieval_results.get("pdf_docs", []),  # Include PDF documents
            "kg_paths": retrieval_results.get("kg_paths", []),
            "kg_insights": retrieval_results.get("kg_insights", []),
            "prompt": retrieval_results.get("prompt", [])  # Include prompt guidance
        }
        
        logger.info(f"Generating response with {len(generator_input['qdrant_docs'])} vector docs, "
                   f"{len(generator_input['pdf_docs'])} PDF docs, and "
                   f"{len(generator_input['kg_insights'])} KG insights")
        
        result = run_rag_generator(generator_input)
        
        logger.info("Response generation completed successfully")
        
        # Ensure we have the minimum structure for the frontend
        if isinstance(result, dict) and "structured" not in result and "conversational" not in result:
            # Add some basic structure
            result = {
                "structured": {
                    "data": {
                        "module1": {
                            "overview": result.get("response", "No detailed response available.")
                        }
                    }
                },
                "conversational": {
                    "data": result.get("response", result.get("text", str(result)))
                }
            }
        
        return {
            "generated_response": result,
            "status": "complete"
        }
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return {
            "errors": [f"Generation error: {str(e)}"],
            "status": "error"
        }

def should_continue(state: GraphState) -> str:
    """Determine the next step in the graph based on current state"""
    # Check for errors first
    if state.get("errors"):
        return "error"
        
    # For clarification node, check if we need clarification
    if "clarification_question" in state and state["clarification_question"]:
        return "needs_clarification"
    
    # Based on what's in the state, determine the appropriate "continue" key
    if state.get("intake_state") and not state.get("refined_query"):
        return "continue"  # After intake, go to clarification
    elif state.get("refined_query") and not state.get("retrieval_results"):
        return "continue_to_retrieve"  # After refine, go to retrieve
    elif state.get("retrieval_results") and not state.get("generated_response"):
        return "continue_to_generate"  # After retrieve, go to generate
    elif state.get("generated_response"):
        return "end"  # After generate, end the flow
    
    # Default fallback if nothing matches
    return "continue"  # Use the most general continue case

def build_graph() -> StateGraph:
    """Build the complete processing graph"""
    builder = StateGraph(GraphState)
    
    # Add all nodes
    builder.add_node("intake", intake_node)
    builder.add_node("clarification", clarification_node_wrapper)
    builder.add_node("refine", refine_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    
    # Set entry point
    builder.set_entry_point("intake")
    
    # Define the flow with improved error handling
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
            "continue": "refine",  # Changed from "continue_to_refine" to "continue"
            # Add a fallback for unexpected values
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
        
        # Get the original query from metadata
        original_query = metadata.get("original_query", "")
        
        # Update metadata with the clarification answer
        updated_metadata = process_clarification_answer(original_query, clarification_answer, metadata or {})
        
        # Make sure we have the original query in metadata
        if "original_query" not in updated_metadata and original_query:
            updated_metadata["original_query"] = original_query
            
        # Create a new graph for complete processing (don't try to skip steps)
        graph = build_graph()
        
        # Start fresh with the complete graph, but include all necessary context
        # in metadata rather than trying to skip nodes
        result = graph.invoke({
            "query": original_query,  # Use the original query
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

# Test function for debugging
def test_graph():
    """Test the graph with a sample query"""
    query = "Tell me about business models"
    print(f"Testing query: {query}")
    
    result = process_query(query)
    
    if result.get("status") == "clarification_needed":
        print(f"Clarification needed: {result.get('clarification_question')}")
        
        # Simulate answering the clarification
        answer = "I want to know about SaaS business models"
        print(f"Answering: {answer}")
        
        # Fix: Pass metadata as a dictionary containing result data
        metadata = {
            "original_query": query,
            "clarification_question": result.get("clarification_question")
        }
        
        final_result = process_clarification("test", answer, metadata)
        
        if final_result.get("generated_response"):
            print("✅ Success! Generated response received.")
        else:
            print("❌ Failed to generate response")
            print("Errors:", final_result.get("errors"))
    
    elif result.get("generated_response"):
        print("✅ Success! No clarification needed.")
    
    else:
        print("❌ Failed")
        print("Errors:", result.get("errors"))

if __name__ == "__main__":
    test_graph()