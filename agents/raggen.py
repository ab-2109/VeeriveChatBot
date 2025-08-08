import os
import hashlib
import logging
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
import re
from langchain_core.messages import AIMessage, BaseMessage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('raggen_agent')



# Load environment variables
load_dotenv()

# === Initialize LLM ===
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

llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
# Use a separate LLM instance for structured output to ensure correct behavior
llm_struct = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY")) \
    .with_structured_output(FullRAGResponse)

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
    """Deduplicate and budget context; supports posts + PDF docs. Caps total size to ~12k chars."""
    if memory_texts is None:
        memory_texts = set()
    seen_hashes: Set[str] = set()
    context: List[str] = []
    total_budget = 12000
    per_chunk_cap = 1500
    used = 0

    for chunk in chunks or []:
        text = ""
        if hasattr(chunk, 'payload'):
            payload = chunk.payload or {}
            text = (payload.get("chunk", "") or
                    payload.get("chunk_text", "") or
                    payload.get("summary", "") or
                    payload.get("content", ""))
        elif isinstance(chunk, dict):
            text = (chunk.get("text", "") or
                    chunk.get("chunk_text", "") or
                    chunk.get("formatted_content", "") or
                    chunk.get("content", "")).strip()
        else:
            text = str(chunk or "").strip()
        if not text:
            continue
        text = text[:per_chunk_cap]
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_hashes or h in memory_texts:
            continue
        if used + len(text) + 5 > total_budget:
            break
        context.append(text)
        used += len(text) + 5
        seen_hashes.add(h)
        memory_texts.add(h)
    return "\n\n---\n\n".join(context), memory_texts


def format_pdf_content(pdf_docs: List[Dict[str, Any]]) -> str:
    """Format PDF document content for inclusion in the prompt"""
    if not pdf_docs:
        return ""
    
    formatted_content = "\n\n=== RELEVANT PDF DOCUMENTS ===\n\n"
    
    # Group by document
    docs_by_id = {}
    for chunk in pdf_docs:
        doc_id = chunk.get('id', 'unknown')
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = {
                'title': chunk.get('title', 'Unknown Document'),
                'chunks': [],
                'source_url': chunk.get('source_url', '')
            }
        
        chunk_text = chunk.get('text', chunk.get('chunk_text', ''))
        if chunk_text:
            chunk_info = f"Content: {chunk_text}"
            if chunk.get('is_table', False):
                chunk_info = f"Table Data: {chunk_text}"
            docs_by_id[doc_id]['chunks'].append(chunk_info)
    
    # Format each document
    for doc_id, doc_info in docs_by_id.items():
        formatted_content += f"**Document: {doc_info['title']}**\n"
        if doc_info['source_url']:
            formatted_content += f"Source: {doc_info['source_url']}\n"
        formatted_content += "\n"
        
        for i, chunk in enumerate(doc_info['chunks'][:3], 1):  # Limit to 3 chunks per doc
            formatted_content += f"Excerpt {i}: {chunk}\n\n"
        
        if len(doc_info['chunks']) > 3:
            formatted_content += f"... and {len(doc_info['chunks']) - 3} more excerpts\n\n"
        
        formatted_content += "---\n\n"
    
    return formatted_content

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
    Extract custom prompt from retrieval agent with enhanced support for PromptSearcher results
    
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
                logger.info("Using custom prompt string")
                return custom_prompt
                
            # Case 2: Empty or no results
            if not custom_prompt or (isinstance(custom_prompt, list) and len(custom_prompt) == 0):
                logger.info("No custom prompt available")
                return None
                
            # Case 3: Dictionary format with 'content' field
            if isinstance(custom_prompt, dict) and "content" in custom_prompt:
                logger.info("Using custom prompt from dictionary")
                return custom_prompt["content"]
                
            # Case 4: List of dictionaries with 'prompt' field (PromptSearcher format)
            if isinstance(custom_prompt, list) and len(custom_prompt) > 0:
                if isinstance(custom_prompt[0], dict) and "prompt" in custom_prompt[0]:
                    prompt_text = custom_prompt[0]["prompt"]
                    if prompt_text and isinstance(prompt_text, str):
                        title = custom_prompt[0].get('title', 'Untitled')
                        sector = custom_prompt[0].get('sector', '')
                        subsector = custom_prompt[0].get('subsector', '')
                        score = custom_prompt[0].get('score', 0)
                        
                        logger.info(f"Using custom prompt: {title} (sector: {sector}, subsector: {subsector}, score: {score})")
                        return prompt_text
                        
            # Case 5: List of strings
            if isinstance(custom_prompt, list) and len(custom_prompt) > 0:
                if isinstance(custom_prompt[0], str) and len(custom_prompt[0].strip()) > 10:
                    logger.info("Using first prompt from list of strings")
                    return custom_prompt[0]
    except Exception as e:
        logger.error(f"Error processing custom prompt: {str(e)}")
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
           Try for 200 lines of content must be generated for each module. More is appreciated.
           Try providing citations for all the facts and data you provide.

Respond in JSON as per the specified schema.
"""

default_convo_system_message = "You are a finance expert analyst. Answer the query in a detailed, structured format. Answer in form of a analysis report, include tables if needed. Explain everything but remain grounded. Always included references and citations for all the facts and data you provide. Try explaining the concepts in a way that is easy to understand for a non-expert."

# === RAG Chain Builders ===

def build_structured_rag_chain():
    """Builds the chain for generating a structured JSON response."""
    format_instructions = parser.get_format_instructions()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        MessagesPlaceholder("history"),
        ("human", "{question}\n\n=== QDRANT CONTEXT ===\n{chunks}\n\n{pdf_content}\n\n=== CITABLE GRAPH PATHS ===\n{paths}\n\nRespond in JSON as per this schema:\n{format_instructions}")
    ])

    return (
        {
            "input_data": RunnablePassthrough()
        }
        | RunnableLambda(lambda data: {
            "question": data["input_data"].get("refined_query", {}).get("refined_query", "Query not found"),
            "chunks": dedupe_chunks(data["input_data"].get("qdrant_docs", []), set())[0],
            "pdf_content": format_pdf_content(data["input_data"].get("pdf_docs", [])),
            "paths": format_kg_paths(data["input_data"].get("kg_paths", [])),
            "system_message": get_custom_prompt(data["input_data"]) or structured_system_message,
            "format_instructions": format_instructions,
            "history": [] # Placeholder for RunnableWithMessageHistory
        })
        | prompt_template
        | llm_struct
        | RunnableLambda(lambda pydantic_obj: pydantic_obj.dict())
    )

def build_convo_rag_chain():
    """Builds the chain for generating a conversational text response."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        MessagesPlaceholder("history"),
        ("human", "{question}\n\nRelevant Chunks:\n{chunks}\n\n{pdf_content}\n\nGraph Paths:\n{paths}")
    ])

    return (
        {
            "input_data": RunnablePassthrough()
        }
        | RunnableLambda(lambda data: {
            "question": data["input_data"].get("refined_query", {}).get("refined_query", "Query not found"),
            "chunks": dedupe_chunks(data["input_data"].get("qdrant_docs", []), set())[0],
            "pdf_content": format_pdf_content(data["input_data"].get("pdf_docs", [])),
            "paths": format_kg_paths(data["input_data"].get("kg_paths", [])),
            "system_message": get_custom_prompt(data["input_data"]) or default_convo_system_message,
            "history": [] # Placeholder for RunnableWithMessageHistory
        })
        | prompt_template
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
    input_messages_key="input_data", # The whole dict is the input
    history_messages_key="history"
)


def generate_structured_response(input_data: Dict[str, Any], session_id: str = "default"):
    """Generates a structured response with robust error handling."""
    try:
        qdrant_count = len(input_data.get("qdrant_docs", []))
        pdf_count = len(input_data.get("pdf_docs", []))
        logger.info(f"Generating structured response with {qdrant_count} vector docs and {pdf_count} PDF docs.")

        # The chain now directly returns a dictionary or raises an error.
        result = structured_chain_with_memory.invoke(input_data, config={"configurable": {"session_id": session_id}})
        
        if isinstance(result, dict):
            return {"status": "success", "data": result}
        
        # This case should ideally not be hit if the chain is correct.
        raise TypeError(f"Expected a dictionary from the structured chain, but got {type(result)}")

    except Exception as e:
        logger.error(f"Structured response generation failed: {e}", exc_info=True)
        return {"status": "error", "data": None, "message": f"Structured generation failed: {str(e)}"}


def generate_conversational_response(input_data: Dict[str, Any], session_id: str = "default"):
    """Generate conversational response. The chain now correctly returns a string."""
    try:
        logger.info(f"Generating conversational response for session: {session_id}")
        
        # The conversational chain now correctly returns a string.
        result_text = convo_chain_with_memory.invoke(input_data, config={"configurable": {"session_id": session_id}})
        
        logger.info("Conversational response generated successfully.")
        return {"status": "success", "data": result_text}
    except Exception as e:
        logger.error(f"Conversational generation error: {str(e)}", exc_info=True)
        return {"status": "error", "data": "Failed to generate response.", "message": str(e)}

def run_rag_generator(input_data: Dict[str, Any], session_id: str = "default"):
    """Generate both structured and conversational responses with enhanced logging and error handling."""
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a string")
    
    logger.info(f"Starting RAG generation for session: {session_id}")
    
    # Log input statistics
    query = input_data.get("refined_query", {}).get("refined_query", "Unknown query")
    qdrant_count = len(input_data.get("qdrant_docs", []))
    pdf_count = len(input_data.get("pdf_docs", []))
    kg_paths_count = len(input_data.get("kg_paths", []))
    kg_insights_count = len(input_data.get("kg_insights", []))
    prompt_count = len(input_data.get("prompt", []) if isinstance(input_data.get("prompt"), list) else 0)
    
    logger.info(f"Query: {query[:100]}...")
    logger.info(f"Resources: {qdrant_count} vector docs, {pdf_count} PDFs, {kg_paths_count} KG paths, "
               f"{kg_insights_count} KG insights, {prompt_count} custom prompts")
    
    # Generate both response types
    structured_response = generate_structured_response(input_data, session_id)
    convo_response = generate_conversational_response(input_data, session_id)
    
    # PDF content for frontend - format PDF content separately to ensure it's available
    pdf_content = format_pdf_content(input_data.get("pdf_docs", []))
    
    # Create a combined result with safe defaults for all fields
    result = {
        "structured": {
            "status": structured_response.get("status", "error"),
            "data": structured_response.get("data", {}),
            "error": structured_response.get("message", None) if structured_response.get("status", "") == "error" else None
        },
        "conversational": {
            "status": convo_response.get("status", "error"),
            "data": convo_response.get("data", "No response generated"),
            "error": convo_response.get("message", None) if convo_response.get("status", "") == "error" else None
        },
        "pdf_content": pdf_content
    }
    
    # Log any errors that occurred
    if structured_response.get("status", "") == "error":
        logger.warning(f"Structured output failed: {structured_response.get('message', 'Unknown error')}")
    if convo_response.get("status", "") == "error":
        logger.warning(f"Conversational output failed: {convo_response.get('message', 'Unknown error')}")
    
    # Debug log the full response structure (truncated for readability)
    logger.info(f"Response structure - structured status: {result['structured']['status']}, " +
                f"conversational status: {result['conversational']['status']}, " +
                f"pdf_content length: {len(result['pdf_content'])}")
    
    # Check if at least one response succeeded
    if (structured_response.get("status", "") == "error" and 
        convo_response.get("status", "") == "error" and
        not pdf_content):
        logger.error("All response types failed")
        raise Exception(f"RAG generation failed: All response types failed.")
    
    logger.info("RAG generation completed successfully")
    return result

if __name__ == "__main__":
    sample_input = {
        "refined_query": {
            "original_query": "What are the business models in BNPL?",
            "refined_query": "What are the dominant business models used by Buy Now Pay Later (BNPL) companies?"
        },
        "qdrant_docs": [],
        "pdf_docs": [],  # Added PDF documents support
        "kg_insights": [],
        "kg_paths": [],
        "prompt": [  # Enhanced prompt structure from PromptSearcher
            {
                "title": "BNPL Business Model Analysis",
                "prompt": "Focus on the financial viability and challenges of each business model. Include recent market changes affecting these models.",
                "sector": "BNPL",
                "subsector": "B2C",
                "score": 0.85
            }
        ]
    }
    result = run_rag_generator(sample_input)
    print(result)