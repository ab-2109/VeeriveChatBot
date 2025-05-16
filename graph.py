from langgraph.graph import StateGraph
from typing import TypedDict, Dict, Any, List, Optional, Annotated
import os
from dotenv import load_dotenv
from datetime import datetime
from agents.intake import process_intake
from agents.refiner import get_refiner
from agents.retrieval import RetrievalAgent
from agents.raggen import run_rag_generator
from agents.clarification import clarification_node, process_clarification_answer

load_dotenv()

class GraphState(TypedDict):
    query: Annotated[str, "replace"]
    metadata: Dict[str, Any]
    intake_state: Optional[Dict[str, Any]]
    refined_query: Optional[Dict[str, Any]]
    retrieval_results: Optional[Dict[str, Any]]
    generated_response: Optional[Dict[str, Any]]
    clarification_question: Optional[str]
    status: Optional[str]
    errors: Annotated[List[str], "append"]

def init_retrieval_agent():
    import urllib.parse
    username = os.getenv("MONGO_USERNAME")
    password = urllib.parse.quote_plus(os.getenv("MONGO_PASSWORD"))
    mongo_uri = f"mongodb+srv://{username}:{password}@veerive.tta8g.mongodb.net/"
    return RetrievalAgent(
        mongo_uri=mongo_uri,
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_key=os.getenv("QDRANT_API"),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USERNAME"),
        neo4j_pass=os.getenv("NEO4J_PASSWORD")
    )

def intake_node(state: GraphState, interactive_callback=None) -> GraphState:
    try:
        query = state["query"]
        metadata = state["metadata"]

        result = process_intake({
            "query": query,
            "metadata": metadata
        }, interactive=False)  # Always use False for interactive to avoid terminal prompts

        if result["status"] == "error":
            return {**state, "errors": [f"Intake error: {result['message']}"]}

        intake_state = result["data"]
        
        # No clarification check here - handled by clarification_node
        return {**state, "intake_state": intake_state}
    except Exception as e:
        return {**state, "errors": [f"Intake processing failed: {str(e)}"]}

def refine_node(state: GraphState) -> GraphState:
    try:
        intake_state = state.get("intake_state", {})
        refined = get_refiner().refine(intake_state)
        if "error" in refined:
            return {**state, "errors": state.get("errors", []) + [refined["error"]]}
        return {**state, "refined_query": refined}
    except Exception as e:
        return {**state, "errors": state.get("errors", []) + [f"Refine error: {str(e)}"]}

def retrieve_node(state: GraphState) -> GraphState:
    try:
        refined_query = state.get("refined_query", {})
        retrieval_agent = init_retrieval_agent()
        results = retrieval_agent.retrieve(refined_query)
        return {**state, "retrieval_results": results}
    except Exception as e:
        return {**state, "errors": state.get("errors", []) + [f"Retrieval error: {str(e)}"]}

def generate_node(state: GraphState) -> GraphState:
    try:
        result = run_rag_generator({
            "refined_query": state.get("refined_query", {"refined_query": state["query"]}),
            "qdrant_docs": state.get("retrieval_results", {}).get("qdrant_docs", []),
            "kg_paths": state.get("retrieval_results", {}).get("kg_paths", []),
            "kg_insights": state.get("retrieval_results", {}).get("kg_insights", [])
        })
        return {**state, "generated_response": result}
    except Exception as e:
        return {**state, "errors": state.get("errors", []) + [f"Generation error: {str(e)}"]}

def has_errors(state: GraphState) -> str:
    return "error" if state.get("errors") else "continue"

def build_graph(interactive_callback=None) -> StateGraph:
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("intake", intake_node)
    builder.add_node("clarification", lambda state: clarification_node(state, interactive_callback))
    builder.add_node("refine", refine_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("END", lambda state: {**state, "status": "completed"})
    # Add a new endpoint specifically for clarification
    builder.add_node("CLARIFY_END", lambda state: state)  # Keep the state as-is

    # Define status router function
    def status_router(state):
        if state.get("errors"):
            return "error"
        elif state.get("status") == "clarification_needed":
            return "needs_clarification"
        else:
            return "continue"

    # Flow with clarification node and conditional routing
    builder.add_edge("intake", "clarification")
    
    # Route clarification to CLARIFY_END instead of END
    builder.add_conditional_edges(
        "clarification",
        status_router,
        {
            "error": "END",
            "needs_clarification": "CLARIFY_END",  # Use the new endpoint
            "continue": "refine"
        }
    )
    
    builder.add_edge("refine", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "END")

    # Add conditional edges for other nodes
    for node in ["intake", "refine", "retrieve", "generate"]:
        builder.add_conditional_edges(
            node, 
            has_errors, 
            {"error": "END", "continue": next_node(node)}
        )

    builder.set_entry_point("intake")
    builder.set_finish_point("END")
    return builder.compile()

def next_node(node):
    return {
        "intake": "clarification",  # Intake now goes to clarification
        "clarification": "refine",   # Clarification goes to refine
        "refine": "retrieve",
        "retrieve": "generate",
        "generate": "END"
    }[node]

def process_query(query: str, metadata: Dict[str, Any] = None, interactive_callback=None) -> Dict[str, Any]:
    # Create a simplified callback wrapper to avoid multiple state updates
    def callback_wrapper(question):
        return lambda current_state: {
            **current_state,
            "status": "clarification_needed",
            "clarification_question": question
        }

    
    # Use the wrapper when building the graph
    graph = build_graph(callback_wrapper)
    
    # Always include an empty errors list to avoid None errors
    result = graph.invoke({
        "query": query,
        "metadata": metadata or {},
        "errors": []
    })
    
    return result

def process_clarification(session_id: str, clarification_answer: str, query: str, metadata: Dict[str, Any], interactive_callback=None) -> Dict[str, Any]:
    # Process the clarification answer
    try:
        # Update metadata with the clarification answer
        updated_metadata = process_clarification_answer(query, clarification_answer, metadata)
        
        # Process the query with updated metadata (including clarifications)
        return process_query(query, updated_metadata, interactive_callback)
    except Exception as e:
        return {
            "status": "error", 
            "errors": [f"Failed to process clarification: {str(e)}"]
        }

if __name__ == "__main__":
    # Example usage with interactive mode
    query = input("Enter your query: ")
    
    # Use a simple mock callback for testing in the terminal
    def mock_interactive_callback(question):
        print(f"\nCLARIFICATION NEEDED: {question}")
        answer = input("Your answer: ")
        return {
            "status": "clarification_needed",
            "clarification_question": question,
            "session_id": "test-session",
            "clarification_answer": answer
        }
    
    result = process_query(query, interactive_callback=mock_interactive_callback)
    
    if result.get("errors"):
        print(f"Errors occurred: {result['errors']}")
    elif "generated_response" in result:
        print("\n✅ Generated Response:")
        response = result["generated_response"]
        response = result["generated_response"]

        if response["structured"]["status"] == "success":
            structured_data = response["structured"]["data"]

            print("\n=== Business Models ===")
            print(structured_data.module1.overview)

            # Display table if available
            table = structured_data.module1.model_comparison
            if table.headers and table.rows:
                print("\n=== Comparison Table ===")
                print(" | ".join(table.headers))
                print("-" * (sum(len(h) for h in table.headers) + 3 * (len(table.headers) - 1)))
                for row in table.rows:
                    print(" | ".join(row))

            print("\n=== Strategic Shifts ===")
            for event in structured_data.module2.major_events:
                print(f"- {event}")

            print("\n=== Key Trends ===")
            for trend in structured_data.module3.key_trends:
                print(f"- {trend}")
        else:
            print("Structured response failed:", response["structured"]["error"])
        if response["conversational"]["status"] == "success":
            conversational_data = response["conversational"]["data"]
            print("\n=== Conversational Response ===")
            print(conversational_data)
    else:
        print("No generated response found.")

        # ...
