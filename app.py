from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from fastapi.middleware.cors import CORSMiddleware
from graph import process_query, process_clarification
import uuid
import traceback

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
    type: str  # 'message', 'clarification', etc.

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    status: str
    session_id: str
    result: Optional[Any] = None
    clarification_question: Optional[str] = None
    conversation_history: Optional[List[ConversationMessage]] = None
    error: Optional[str] = None

# === Single POST endpoint ===
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if request.session_id:
        return await handle_clarification(request)
    else:
        return await handle_new_query(request)

async def handle_new_query(request: QueryRequest) -> QueryResponse:
    """Handle a brand new query"""
    session_id = str(uuid.uuid4())
    conversation_history = [
        ConversationMessage(role="user", content=request.query, type="message")
    ]
    try:
        result = process_query(request.query, request.metadata or {})
        # Check if clarification is needed
        if result.get("status") == "clarification_needed":
            active_sessions[session_id] = {
                "graph_state": result,
                "conversation_history": conversation_history,
                "original_query": request.query,
                "metadata": request.metadata or {},
                "clarification_question": result.get("clarification_question", "")
            }
            conversation_history.append(
                ConversationMessage(
                    role="assistant",
                    content=result.get("clarification_question", "Please clarify your question."),
                    type="clarification"
                )
            )
            return QueryResponse(
                status="clarification_needed",
                session_id=session_id,
                clarification_question=result.get("clarification_question", "Please clarify your question."),
                conversation_history=conversation_history
            )
        # Check for errors
        if result.get("errors"):
            conversation_history.append(
                ConversationMessage(
                    role="assistant",
                    content=f"I encountered an error: {'; '.join(result['errors'])}",
                    type="message"
                )
            )
            return QueryResponse(
                status="error",
                session_id=session_id,
                error="; ".join(result["errors"]),
                conversation_history=conversation_history
            )
        # Success - add response to conversation
        response_content = format_response(result.get("generated_response", result))
        conversation_history.append(
            ConversationMessage(
                role="assistant",
                content=response_content,
                type="message"
            )
        )
        return QueryResponse(
            status="complete",
            session_id=session_id,
            result=result.get("generated_response", result),
            conversation_history=conversation_history
        )
    except Exception as e:
        print(f"New query error: {str(e)}")
        print(traceback.format_exc())
        return QueryResponse(
            status="error", 
            session_id=session_id,
            error=str(e)
        )

async def handle_clarification(request: QueryRequest) -> QueryResponse:
    """Handle clarification responses"""
    session_id = request.session_id
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        conversation_history = session.get("conversation_history", [])
        conversation_history.append(
            ConversationMessage(
                role="user",
                content=request.query,
                type="message"
            )
        )
        metadata = session.get("metadata", {})
        if "original_query" not in metadata:
            metadata["original_query"] = session.get("original_query", "")
        metadata["trigger_clarification"] = False
        metadata["clarified"] = True
        previous_question = session.get("clarification_question", "")
        if "clarifications" not in metadata:
            metadata["clarifications"] = []
        metadata["clarifications"].append({
            "question": previous_question,
            "answer": request.query
        })
        result = process_clarification(
            session_id=session_id,
            clarification_answer=request.query,
            metadata=metadata
        )
        if result.get("status") == "clarification_needed":
            active_sessions[request.session_id]["graph_state"] = result
            active_sessions[request.session_id]["conversation_history"] = conversation_history
            conversation_history.append(
                ConversationMessage(
                    role="assistant",
                    content=result.get("clarification_question"),
                    type="clarification"
                )
            )
            return QueryResponse(
                status="clarification_needed",
                session_id=request.session_id,
                clarification_question=result.get("clarification_question"),
                conversation_history=conversation_history
            )
        del active_sessions[request.session_id]
        if result.get("errors"):
            conversation_history.append(
                ConversationMessage(
                    role="assistant",
                    content=f"I encountered an error: {'; '.join(result['errors'])}",
                    type="message"
                )
            )
            return QueryResponse(
                status="error",
                session_id=request.session_id,
                error="; ".join(result["errors"]),
                conversation_history=conversation_history
            )
        response_content = format_response(result.get("generated_response", result))
        conversation_history.append(
            ConversationMessage(
                role="assistant",
                content=response_content,
                type="message"
            )
        )
        return QueryResponse(
            status="complete",
            session_id=request.session_id,
            result=result.get("generated_response", result),
            conversation_history=conversation_history
        )
    except Exception as e:
        print(f"Clarification error: {str(e)}")
        print(traceback.format_exc())
        if request.session_id in active_sessions:
            del active_sessions[request.session_id]
        return QueryResponse(
            status="error",
            session_id=request.session_id,
            error=str(e)
        )

def format_response(response_data: Dict[str, Any]) -> str:
    """Format the response data into a JSON string for the frontend to parse and render tables, PDFs, etc."""
    import json
    if not response_data:
        return json.dumps({"error": "I couldn't generate a response."})
    # Always return as JSON string for frontend parsing
    try:
        return json.dumps(response_data, ensure_ascii=False)
    except Exception:
        return str(response_data)

# === Additional endpoints for convenience ===
@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "status": session["graph_state"].get("status", "unknown"),
        "conversation_length": len(session.get("conversation_history", [])),
        "awaiting_clarification": session["graph_state"].get("status") == "clarification_needed"
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": "Session deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

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