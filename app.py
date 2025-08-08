from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from fastapi.middleware.cors import CORSMiddleware
from graph import process_query, process_clarification
import uuid
import traceback
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
active_sessions = {}

class ConversationMessage(BaseModel):
    role: str
    content: str
    type: str = "message"  # 'message', 'clarification', etc.
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ClarificationRequest(BaseModel):
    session_id: str
    answer: str

class QueryResponse(BaseModel):
    status: str
    session_id: str
    result: Optional[Any] = None
    clarification_question: Optional[str] = None
    conversation_history: Optional[List[ConversationMessage]] = None
    error: Optional[str] = None

# --- Helper functions ---


def format_response(response_data: Dict[str, Any], session_id: str = "") -> Dict[str, Any]:
    """Format the response from the graph for the frontend, including table-aware PDF content."""
    try:
        # If clarification needed, short-circuit
        if response_data.get("status") == "clarification_needed":
            return {
                "status": "clarification_needed",
                "clarification_question": response_data.get("clarification_question", ""),
                "session_id": session_id
            }

        # Helper to split pdf_docs into tables/texts
        def split_pdf_docs(pdf_docs):
            tables, texts = [], []
            for d in pdf_docs or []:
                content = d.get("formatted_content") or d.get("text") or d.get("chunk_text") or ""
                entry = {
                    "id": d.get("id"),
                    "title": d.get("title", "Unknown PDF"),
                    "content": content,
                    "is_table": bool(d.get("is_table", False)),
                    "source_url": d.get("source_url", ""),
                    "score": d.get("score", 0),
                    "relevance_score": d.get("relevance_score", 0),
                }
                if entry["is_table"]:
                    tables.append(entry)
                else:
                    texts.append(entry)
            return tables, texts

        # Extract generated_response if present
        if "generated_response" in response_data:
            response = response_data["generated_response"]

            # Attempt to extract pdf content from multiple places
            pdf_docs = response_data.get("retrieval_results", {}).get("pdf_docs") or                        response_data.get("pdf_docs") or                        response.get("pdf_docs") or []

            tables, texts = split_pdf_docs(pdf_docs)

            # Conversational content
            conversational_data = ""
            if isinstance(response, dict) and "conversational" in response:
                # Might be plain string or dict with 'data'
                conv = response["conversational"]
                conversational_data = conv.get("data") if isinstance(conv, dict) else conv

            # Structured content
            structured_data = {}
            if isinstance(response, dict) and "structured" in response:
                sd = response["structured"]
                structured_data = sd.get("data") if isinstance(sd, dict) and "data" in sd else sd

            # Combined markdown for PDFs (for legacy UI that expects a single string)
            try:
                from agents.raggen import format_pdf_content as _fp
                combined_pdf_markdown = _fp(pdf_docs) if pdf_docs else ""
            except Exception:
                combined_pdf_markdown = ""

            return {
                "status": "complete",
                "result": {
                    "conversational": conversational_data or "No conversational response available.",
                    "structured": structured_data or {},
                    "pdf": {
                        "tables": tables,
                        "texts": texts,
                        "combined_markdown": combined_pdf_markdown
                    },
                    "pdf_content": combined_pdf_markdown
                },
                "session_id": session_id
            }

        # Handle explicit errors
        if response_data.get("errors"):
            return {
                "status": "error",
                "error": "; ".join(map(str, response_data.get("errors", []))),
                "session_id": session_id
            }

        # Unknown state fallback
        return {
            "status": "unknown",
            "result": {
                "conversational": str(response_data),
                "structured": {},
                "pdf": {"tables": [], "texts": [], "combined_markdown": ""}
            },
            "session_id": session_id
        }

    except Exception as e:
        logger.exception(f"Error formatting response: {str(e)}")
        return {
            "status": "error",
            "error": f"Error formatting response: {str(e)}",
            "session_id": session_id
        }

def extract_conversational_content(response):
    """Extract conversational content from response regardless of format"""
    if isinstance(response, dict):
        if "conversational" in response:
            # New format with structured/conversational split
            content = response["conversational"].get("data", "")
            if isinstance(content, dict):
                return str(content)
            return content
        elif "text" in response:
            # Simple format with text field
            return response["text"]
    
    # Fallback to string representation
    return str(response)

# --- API endpoints ---

@app.get("/")
async def root():
    return {
        "message": "Graph API is running",
        "endpoints": {
            "POST /query": "Submit query or clarification response",
            "GET /sessions/{id}": "Check session status",
            "DELETE /sessions/{id}": "Delete session"
        },
        "active_sessions": len(active_sessions),
        "usage": {
            "new_query": "POST /query with {query: 'your question'}",
            "clarification": "POST /query with {query: 'your answer', session_id: 'session_id'}"
        }
    }

@app.post("/query")
async def handle_query(query_request: QueryRequest):
    """Single endpoint for both new queries and clarifications"""
    try:
        query = query_request.query
        session_id = query_request.session_id
        
        logger.info(f"Received query: '{query}' with session_id: {session_id}")
        
        if not query:
            return {"status": "error", "error": "Query cannot be empty"}
        
        # New query if no session_id is provided
        if not session_id:
            logger.info(f"Processing new query: '{query}'")
            result = process_query(query, query_request.metadata or {})
            
            # DEBUG: Print retrieval results to terminal
            if "retrieval_results" in result:
                print("\n========== RETRIEVAL RESULTS ==========")
                print(f"Query: '{query}'")
                retrieval = result["retrieval_results"]
                
                # Print regular documents
                if "qdrant_docs" in retrieval:
                    print(f"\n----- REGULAR DOCUMENTS ({len(retrieval['qdrant_docs'])}) -----")
                    for i, doc in enumerate(retrieval["qdrant_docs"]):
                        print(f"Doc {i+1}. Score: {doc.get('score')}")
                        print(f"Text: {doc.get('text')[:200]}..." if len(doc.get('text', '')) > 200 else doc.get('text'))
                        print("---")
                
                # Print PDF documents
                if "pdf_docs" in retrieval:
                    print(f"\n----- PDF DOCUMENTS ({len(retrieval['pdf_docs'])}) -----")
                    for i, doc in enumerate(retrieval["pdf_docs"]):
                        print(f"PDF {i+1}. Score: {doc.get('score')}, Relevance: {doc.get('relevance_score')}")
                        print(f"Content: {doc.get('formatted_content') or doc.get('text')[:200]}..." if len(doc.get('content', '')) > 200 else doc.get('formatted_content') or doc.get('text'))
                        print("---")
                
                # Print KG insights
                if "kg_insights" in retrieval:
                    print(f"\n----- KNOWLEDGE GRAPH INSIGHTS ({len(retrieval['kg_insights'])}) -----")
                    for i, insight in enumerate(retrieval["kg_insights"]):
                        print(f"Insight {i+1}: {insight}")
                
                # Print KG paths
                if "kg_paths" in retrieval:
                    print(f"\n----- KNOWLEDGE GRAPH PATHS ({len(retrieval['kg_paths'])}) -----")
                    for i, path in enumerate(retrieval["kg_paths"]):
                        print(f"Path {i+1}: {path}")
                
                print("=======================================\n")
            
            session_id = str(uuid.uuid4())
            
            # Store conversation history
            conversation_history = [
                ConversationMessage(
                    role="user",
                    content=query,
                    session_id=session_id,
                    timestamp=datetime.now()
                )
            ]
            
            # Create a new session
            active_sessions[session_id] = {
                "status": result.get("status", "unknown"),
                "query": query,
                "timestamp": datetime.now(),
                "result": result,
                "conversation_history": conversation_history
            }
            
            # Add clarification question to conversation history if needed
            if result.get("status") == "clarification_needed":
                logger.info(f"Clarification needed: {result.get('clarification_question')}")
                conversation_history.append(
                    ConversationMessage(
                        role="assistant",
                        content=result.get("clarification_question", "Please clarify your question."),
                        type="clarification",
                        session_id=session_id,
                        timestamp=datetime.now()
                    )
                )
            
            # Add response to conversation history if available
            elif "generated_response" in result:
                response = result["generated_response"]
                content = extract_conversational_content(response)
                logger.info(f"Generated response: {content[:100]}...")
                
                # DEBUG: Print full generated response
                print("\n========== GENERATED RESPONSE ==========")
                print(f"Query: '{query}'")
                print(f"Response type: {type(response)}")
                
                if isinstance(response, dict):
                    if "structured" in response:
                        print("\n----- STRUCTURED RESPONSE -----")
                        print(response["structured"])
                    
                    if "conversational" in response:
                        print("\n----- CONVERSATIONAL RESPONSE -----")
                        print(response["conversational"])
                else:
                    print("\n----- SIMPLE RESPONSE -----")
                    print(response)
                
                print("=======================================\n")
                
                conversation_history.append(
                    ConversationMessage(
                        role="assistant",
                        content=content,
                        session_id=session_id,
                        timestamp=datetime.now()
                    )
                )
            
            # Format the response
            response = format_response(result, session_id)
            response["conversation_history"] = conversation_history
            return response
        
        # Existing session - handle as clarification
        elif session_id in active_sessions:
            session = active_sessions[session_id]
            logger.info(f"Processing clarification for session {session_id}: '{query}'")
            
            # Get conversation history or create if missing
            if "conversation_history" not in session:
                session["conversation_history"] = []
            
            # Add user message to history
            session["conversation_history"].append(
                ConversationMessage(
                    role="user",
                    content=query,
                    session_id=session_id,
                    timestamp=datetime.now()
                )
            )
            
            # Prepare metadata for clarification
            metadata = session.get("metadata", {})
            if "original_query" not in metadata:
                metadata["original_query"] = session.get("query", "")
            metadata["clarification_question"] = session.get("result", {}).get("clarification_question", "")
            
            # Process the clarification
            result = process_clarification(session_id, query, metadata)
            
            # DEBUG: Print clarification processing results
            print("\n========== CLARIFICATION RESULTS ==========")
            print(f"Original query: '{metadata.get('original_query')}'")
            print(f"Clarification: '{query}'")
            
            if "retrieval_results" in result:
                retrieval = result["retrieval_results"]
                
                # Print regular documents
                if "qdrant_docs" in retrieval:
                    print(f"\n----- REGULAR DOCUMENTS ({len(retrieval['qdrant_docs'])}) -----")
                    for i, doc in enumerate(retrieval["qdrant_docs"]):
                        print(f"Doc {i+1}. Score: {doc.get('score')}")
                        print(f"Text: {doc.get('text')[:200]}..." if len(doc.get('text', '')) > 200 else doc.get('text'))
                        print("---")
                
                # Print PDF documents
                if "pdf_docs" in retrieval:
                    print(f"\n----- PDF DOCUMENTS ({len(retrieval['pdf_docs'])}) -----")
                    for i, doc in enumerate(retrieval["pdf_docs"]):
                        print(f"PDF {i+1}. Score: {doc.get('score')}, Relevance: {doc.get('relevance_score')}")
                        print(f"Content: {doc.get('formatted_content') or doc.get('text')[:200]}..." if len(doc.get('content', '')) > 200 else doc.get('formatted_content') or doc.get('text'))
                        print("---")
            
            print("==========================================\n")
            
            # Update session
            session["result"] = result
            session["status"] = result.get("status", "unknown")
            session["timestamp"] = datetime.now()
            
            # Add assistant response to conversation history
            if "generated_response" in result:
                response = result["generated_response"]
                content = extract_conversational_content(response)
                logger.info(f"Generated clarification response: {content[:100]}...")
                
                # DEBUG: Print full generated response
                print("\n========== CLARIFICATION RESPONSE ==========")
                print(f"Response type: {type(response)}")
                
                if isinstance(response, dict):
                    if "structured" in response:
                        print("\n----- STRUCTURED RESPONSE -----")
                        print(response["structured"])
                    
                    if "conversational" in response:
                        print("\n----- CONVERSATIONAL RESPONSE -----")
                        print(response["conversational"])
                else:
                    print("\n----- SIMPLE RESPONSE -----")
                    print(response)
                
                print("============================================\n")
                
                session["conversation_history"].append(
                    ConversationMessage(
                        role="assistant",
                        content=content,
                        session_id=session_id,
                        timestamp=datetime.now()
                    )
                )
            elif result.get("status") == "clarification_needed":
                logger.info(f"Additional clarification needed: {result.get('clarification_question')}")
                session["conversation_history"].append(
                    ConversationMessage(
                        role="assistant",
                        content=result.get("clarification_question", "Please clarify further."),
                        type="clarification",
                        session_id=session_id,
                        timestamp=datetime.now()
                    )
                )
            
            # Format the response
            response = format_response(result, session_id)
            response["conversation_history"] = session["conversation_history"]
            return response
        
        # Invalid session
        else:
            logger.warning(f"Invalid session ID: {session_id}")
            return {
                "status": "error", 
                "error": "Invalid session ID. Session may have expired or been deleted."
            }
            
    except Exception as e:
        logger.exception(f"Error processing query: {str(e)}")
        traceback.print_exc()  # Print full stack trace to terminal
        return {"status": "error", "error": str(e)}

@app.post("/clarification")
async def handle_clarification(clarification: ClarificationRequest):
    """Legacy endpoint for clarification responses from the frontend"""
    try:
        session_id = clarification.session_id
        answer = clarification.answer
        
        if session_id not in active_sessions:
            return {"status": "error", "error": "Invalid session ID"}
        
        # Convert clarification request to a query request and delegate
        query_request = QueryRequest(
            query=answer,
            session_id=session_id
        )
        
        return await handle_query(query_request)
        
    except Exception as e:
        logger.exception(f"Clarification error: {str(e)}")
        return {"status": "error", "error": f"Error processing clarification: {str(e)}"}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "status": session.get("status", "unknown"),
        "conversation_length": len(session.get("conversation_history", [])),
        "awaiting_clarification": session.get("status") == "clarification_needed"
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": "Session deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")