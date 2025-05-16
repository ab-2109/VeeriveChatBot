import os
import hashlib
from typing import List, Dict, Any, Optional, Set
import langchain.callbacks
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import langchain
from langchain.callbacks.manager import CallbackManager



# Load environment variables
load_dotenv()

# === Initialize LLM ===
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

# === Define Structured Output Format ===
class TableData(BaseModel):
    headers: List[str]
    rows: List[List[str]]

class Module1BusinessModel(BaseModel):
    overview: str
    model_details: List[str]
    model_comparison: TableData
    examples: List[str]

class Module2StrategyInnovation(BaseModel):
    major_events: List[str]
    expert_opinions: List[str]

class Module3TrendAnalyzer(BaseModel):
    key_trends: List[str]
    associated_themes: List[str]

class Module4GlobalTrends(BaseModel):
    global_events: List[str]
    global_model_shifts: List[str]

class FullRAGResponse(BaseModel):
    module1: Module1BusinessModel = Field(..., description="Detailed breakdown of business models")
    module2: Module2StrategyInnovation = Field(..., description="Strategy & innovation insights")
    module3: Module3TrendAnalyzer = Field(..., description="Trend analysis in the industry, sector or subsector")
    module4: Module4GlobalTrends = Field(..., description="International Analysis and global trends")

# Initialize output parser
parser = PydanticOutputParser(pydantic_object=FullRAGResponse)

# === Conversational Memory Setup ===
convo_buffers: Dict[str, ChatMessageHistory] = {}

def get_user_convo_history(session_id: str) -> ChatMessageHistory:
    if session_id not in convo_buffers:
        convo_buffers[session_id] = ChatMessageHistory()
    return convo_buffers[session_id]

# === Helper Functions ===
def dedupe_chunks(chunks: List[Any], memory_texts: Optional[Set[str]] = None) -> tuple[str, Set[str]]:
    if memory_texts is None:
        memory_texts = set()
    seen_hashes = set()
    context = []
    for chunk in chunks:
        if hasattr(chunk, 'payload'):
            text = chunk.payload.get("chunk", "") or chunk.payload.get("summary", "")
        elif isinstance(chunk, dict):
            text = chunk.get("text", "").strip()
        else:
            text = str(chunk).strip()
        if not text:
            continue
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen_hashes and h not in memory_texts:
            context.append(text)
            seen_hashes.add(h)
            memory_texts.add(h)
    return "\n\n---\n\n".join(context), memory_texts

def format_kg_paths(paths: List[Dict[str, Any]]) -> str:
    output = []
    for p in paths:
        if isinstance(p, dict):
            if "path" in p:
                nodes = " → ".join(str(n.get("id", n)) for n in p["path"] if isinstance(n, dict))
                output.append(f"- {nodes}")
            elif "title" in p:
                output.append(f"- [{p['title']}]({p.get('url', '')})")
    return "\n".join(output)

# === Dynamic Prompt Creation ===
def get_custom_prompt(input_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract custom prompt from retrieval agent if available
    
    Args:
        input_data: The input data that might contain a custom prompt
        
    Returns:
        Optional[str]: The custom prompt if available, None otherwise
    """
    try:
        # Look for a custom prompt in the input data structure
        if "prompt" in input_data:
            custom_prompt = input_data.get("prompt")
            
            # Case 1: Direct string prompt
            if isinstance(custom_prompt, str) and len(custom_prompt.strip()) > 10:
                print(f"Using custom prompt string")
                return custom_prompt
                
            # Case 2: Empty or no results
            if not custom_prompt or (isinstance(custom_prompt, list) and len(custom_prompt) == 0):
                print("No custom prompt available")
                return None
                
            # Case 3: Dictionary format with 'content' field
            if isinstance(custom_prompt, dict) and "content" in custom_prompt:
                print(f"Using custom prompt from dictionary")
                return custom_prompt["content"]
                
            # Case 4: List of dictionaries with 'prompt' field
            if isinstance(custom_prompt, list) and len(custom_prompt) > 0:
                if isinstance(custom_prompt[0], dict) and "prompt" in custom_prompt[0]:
                    prompt_text = custom_prompt[0]["prompt"]
                    if prompt_text and isinstance(prompt_text, str):
                        print(f"Using custom prompt: {custom_prompt[0].get('title', 'Untitled')}")
                        return prompt_text
    except Exception as e:
        print(f"Error processing custom prompt: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return None

# === Structured Prompt Template ===
structured_system_message = """
You are a financial research analyst in a Big Professional MNC. You provide detailed analysis. Based on the user's query and the provided context, generate an in-depth response structured into the following modules:

Module 1: Key Concepts and Models  
→ Define the key models or concepts relevant to the topic.  
→ Provide reasoning behind them, differences, pros/cons, and supporting entities.  
→ Cite related sources and graph paths.

Module 2: Strategic Shifts and Innovation  
→ Summarize recent strategy changes, innovations, and pivots.  
→ Cite examples and sources where relevant.

Module 3: Trend Analysis  
→ List down major market or ecosystem trends linked to the topic.  
→ Explain what drives them, supported by document and graph info.

Module 4: Global Comparisons / Broader Context  
→ Provide any global view, alternative models, or macro-level insight.  
→ Include related international entities or events, if present.

IMPORTANT : Use natural language and include tables or bullets if necessary. Cite source titles or entities whenever you use specific facts.
           The output provided by you must be grounded in the context provided. Any fact or data not present in the context is not permissible.
           Atleast 200 lines of content must be generated for each module. More is appreciated.
           Provide citations for all the facts and data you provide.
           Do not return without citations and atleast reaching 200 lines of content.

Respond in JSON as per the specified schema.
"""

default_convo_system_message = "You are a finance expert analyst. Answer the query in a detailed, structured format. Answer in form of a analysis report, include tables if needed. Explain everything but remain grounded."

# === RAG Chain Builders ===
def build_structured_rag_chain():
    format_instructions = parser.get_format_instructions()
    
    return (
        {
            "question": lambda inputs: inputs,  # Pass through the entire input
            "chunks": lambda inputs: inputs.get("qdrant_docs", []),
            "paths": lambda inputs: format_kg_paths(inputs.get("kg_paths", []))
        }
        | RunnableLambda(lambda data: {
            # Extract the refined query
            "question": data["question"].get("refined_query", {}).get("refined_query", 
                      data["question"].get("original_query", "Query not found")),
            "chunks": dedupe_chunks(data["chunks"], data.setdefault("memory_texts", set()))[0],
            "paths": data["paths"],
            "format_instructions": format_instructions,
            "memory_texts": data["memory_texts"],
            "history": [],  # Always provide history
            "system_message": get_custom_prompt(data["question"]) or structured_system_message  # Use custom prompt if available
        })
        | RunnableLambda(lambda data: {
            **data,
            "prompt": ChatPromptTemplate.from_messages([
                ("system", data["system_message"]),
                MessagesPlaceholder("history"),
                ("human", "{question}\n\n=== QDRANT CONTEXT ===\n{chunks}\n\n=== CITABLE GRAPH PATHS ===\n{paths}\n\nRespond in JSON as per this schema:\n{format_instructions}")
            ])
        })
        | RunnableLambda(lambda data: data["prompt"].format(
            question=data["question"],
            chunks=data["chunks"],
            paths=data["paths"],
            format_instructions=data["format_instructions"],
            history=data["history"]
        ))
        | llm
        | StrOutputParser()
    )

def build_convo_rag_chain():
    return (
        {
            "question": lambda inputs: inputs["refined_query"]["refined_query"] if "refined_query" in inputs and "refined_query" in inputs["refined_query"] else "",
            "chunks": lambda inputs: inputs.get("qdrant_docs", []),
            "paths": lambda inputs: format_kg_paths(inputs.get("kg_paths", []))
        }
        | RunnableLambda(lambda data: {
            "question": data["question"],
            "chunks": dedupe_chunks(data["chunks"], data.setdefault("memory_texts", set()))[0],
            "paths": data["paths"],
            "memory_texts": data["memory_texts"],
            "history": [],
            "system_message": get_custom_prompt(data) or default_convo_system_message  # Use custom prompt if available
        })
        | RunnableLambda(lambda data: {
            **data,
            "prompt": ChatPromptTemplate.from_messages([
                ("system", data["system_message"]),
                MessagesPlaceholder("history"),
                ("human", "{question}\n\nRelevant Chunks:\n{chunks}\n\nGraph Paths:\n{paths}")
            ])
        })
        | RunnableLambda(lambda data: data["prompt"].format(
            question=data["question"],
            chunks=data["chunks"],
            paths=data["paths"],
            history=data["history"]
        ))
        | llm
        | StrOutputParser()
    )

structured_rag_chain = build_structured_rag_chain()
convo_rag_chain = build_convo_rag_chain()

structured_chain_with_memory = RunnableWithMessageHistory(
    structured_rag_chain,
    get_session_history=get_user_convo_history,
    input_messages_key="question",
    history_messages_key="history"
)

convo_chain_with_memory = RunnableWithMessageHistory(
    convo_rag_chain,
    get_session_history=get_user_convo_history,
    input_messages_key="question",
    history_messages_key="history"
)

def generate_structured_response(input_data: Dict[str, Any], session_id: str = "default"):
    try:
        result = structured_chain_with_memory.invoke(input_data, config={"configurable": {"session_id": session_id}})
        structured_result = parser.parse(result)
        return {"status": "success", "data": structured_result}
    except Exception as e:
        return {"status": "error", "message": f"Structured parsing error: {str(e)}"}

def generate_conversational_response(input_data: Dict[str, Any], session_id: str = "default"):
    try:
        result = convo_chain_with_memory.invoke(input_data, config={"configurable": {"session_id": session_id}})
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_rag_generator(input_data: Dict[str, Any], session_id: str = "default"):
    """Generate both structured and conversational responses and return them together."""
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a string")
    
    # Generate both response types
    structured_response = generate_structured_response(input_data, session_id)
    convo_response = generate_conversational_response(input_data, session_id)
    # print(f"Structured Response: {structured_response}")
    # print(f"Conversational Response: {convo_response}")
    
    # Create a combined result
    result = {
        "structured": {
            "status": structured_response["status"],
            "data": structured_response.get("data", None),
            "error": structured_response.get("message", None) if structured_response["status"] == "error" else None
        },
        "conversational": {
            "status": convo_response["status"],
            "data": convo_response.get("data", None),
            "error": convo_response.get("message", None) if convo_response["status"] == "error" else None
        }
    }
    
    # Log any errors that occurred
    if structured_response["status"] == "error":
        print(f"Warning: Structured output failed: {structured_response['message']}")
    if convo_response["status"] == "error":
        print(f"Warning: Conversational output failed: {convo_response['message']}")
    
    # Check if at least one response succeeded
    if structured_response["status"] == "error" and convo_response["status"] == "error":
        raise Exception(f"RAG generation failed: Both response types failed.")
    
    return result

if __name__ == "__main__":
    sample_input = {
        "refined_query": {
            "original_query": "What are the business models in BNPL?",
            "refined_query": "What are the dominant business models used by Buy Now Pay Later (BNPL) companies?"
        },
        "qdrant_docs": [],
        "kg_insights": [],
        "kg_paths": [],
        "prompt": "Focus on the financial viability and challenges of each business model. Include recent market changes affecting these models."
    }
    result = run_rag_generator(sample_input)
    print(result)