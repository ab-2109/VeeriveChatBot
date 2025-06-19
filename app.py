from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
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

active_sessions = {}

class QueryRequest(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = None

class ClarificationResponse(BaseModel):
    session_id: str
    answer: str

class QueryResponse(BaseModel):
    status: str
    session_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    clarification_question: Optional[str] = None
    error: Optional[str] = None

def clarification_callback(session_id: str):
    return lambda question: {
        "status": "clarification_needed",
        "clarification_question": question,
        "session_id": session_id
    }

# === POST /graph ===
@app.post("/graph", response_model=QueryResponse)
async def graph_endpoint(request: QueryRequest):
    session_id = str(uuid.uuid4())
    metadata = request.metadata or {}
    metadata["session_id"] = session_id
    metadata["trigger_clarification"] = True  

    try:
        result = process_query(
            query=request.query,
            metadata=metadata,
            interactive_callback=clarification_callback(session_id)
        )

        if result.get("clarification_question") and result.get("status") != "error":
            active_sessions[session_id] = {
                "query": request.query,
                "metadata": metadata,
                "state": "awaiting_clarification",
                "clarification_question": result["clarification_question"]
            }
            
            return QueryResponse(
                status="clarification_needed",
                session_id=session_id,
                clarification_question=result["clarification_question"]
            )
        
        if isinstance(result, dict) and result.get("status") == "clarification_needed":
            active_sessions[session_id] = {
                "query": request.query,
                "metadata": metadata,
                "state": "awaiting_clarification",
                "clarification_question": result["clarification_question"]
            }
            active_sessions[session_id]["metadata"]["last_clarification_question"] = result["clarification_question"]

            return QueryResponse(
                status="clarification_needed",
                session_id=session_id,
                clarification_question=result["clarification_question"]
            )

        return QueryResponse(
            status="complete",
            result=result.get("generated_response", result)
        )
    except Exception as e:
        print(f"Graph endpoint error: {str(e)}")
        print(traceback.format_exc())
        return QueryResponse(status="error", error=str(e))

# === POST /clarify ===
@app.post("/clarify", response_model=QueryResponse)
async def clarification_endpoint(response: ClarificationResponse):
    session = active_sessions.get(response.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        metadata = session["metadata"]

        # Inject clarification question + answer
        last_question = session.get("clarification_question", "Clarification question")
        if "clarifications" not in metadata:
            metadata["clarifications"] = []

        metadata["clarifications"].append({
            "question": last_question,
            "answer": response.answer
        })

        metadata["trigger_clarification"] = False
        metadata["clarified"] = True

        result = process_clarification(
            session_id=response.session_id,
            clarification_answer=response.answer,
            query=session["query"],
            metadata=metadata,
            interactive_callback=clarification_callback(response.session_id)
        )

        if isinstance(result, dict) and result.get("status") == "clarification_needed":
            active_sessions[response.session_id]["state"] = "awaiting_clarification"
            active_sessions[response.session_id]["clarification_question"] = result["clarification_question"]
            metadata["last_clarification_question"] = result["clarification_question"]

            return QueryResponse(
                status="clarification_needed",
                session_id=response.session_id,
                clarification_question=result["clarification_question"]
            )

        del active_sessions[response.session_id]
        return QueryResponse(
            status="complete",
            result=result.get("generated_response", result)
        )

    except Exception as e:
        print(f"Clarification error: {str(e)}")
        print(traceback.format_exc())
        return QueryResponse(status="error", error=str(e))

# === GET / ===
@app.get("/")
async def root():
    return {"message": "Graph API is running. Use POST /graph to query."}
