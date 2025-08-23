import os
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import langchain.callbacks
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from pydantic import BaseModel, Field, create_model
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

# === Dynamic Pydantic Model Creation ===
class TableData(BaseModel):
    headers: List[str]
    rows: List[List[str]]

class DynamicModuleData(BaseModel):
    title: str = Field(..., description="Module title")
    content: str = Field(..., description="Module content")
    bullet_points: List[str] = Field(default_factory=list, description="Key points")
    tables: List[TableData] = Field(default_factory=list, description="Tables if any")
    examples: List[str] = Field(default_factory=list, description="Examples if any")

def create_dynamic_response_model(modules: List[Tuple[str, str]]) -> type:
    """
    Create a dynamic Pydantic model based on the modules found in the prompt.
    
    Args:
        modules: List of (module_title, module_description) tuples
    
    Returns:
        Dynamically created Pydantic model class
    """
    fields = {}
    
    for i, (title, description) in enumerate(modules):
        # Clean the title to create a valid field name
        field_name = f"module_{i+1}"
        clean_title = re.sub(r'[^a-zA-Z0-9_]', '_', title.lower())
        if clean_title and clean_title != 'module':
            field_name = clean_title
        
        # Create field with description from the module content
        fields[field_name] = (DynamicModuleData, Field(..., description=f"{title}: {description[:200]}..."))
    
    # If no modules found, fall back to generic structure
    if not fields:
        fields = {
            "analysis": (DynamicModuleData, Field(..., description="General analysis")),
            "insights": (DynamicModuleData, Field(..., description="Key insights")),
            "recommendations": (DynamicModuleData, Field(..., description="Recommendations"))
        }
    
    # Create the dynamic model
    DynamicResponse = create_model(
        'DynamicRAGResponse',
        **fields,
        __base__=BaseModel
    )
    
    return DynamicResponse

# === Enhanced Module Extraction ===
def extract_modules_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Extract modules and their descriptions from structured text with enhanced pattern matching.
    
    Args:
        text: Text that may contain module definitions
        
    Returns:
        List of (module_title, module_description) tuples
    """
    modules = []
    
    # Enhanced module patterns to catch various formats
    module_patterns = [
        # "Module 1:", "Module 1 —", "Module 1 -", "Module 1 – "
        r'Module\s+(\d+|[A-Z])[:\s—–-]+\s*([^•]*?)(?=\s*(?:Module\s+(?:\d+|[A-Z])|$))',
        # "Module 1: Title" followed by content
        r'Module\s+(\d+|[A-Z])[:\s—–-]+\s*([^\n•]*?)(?:\n([^M]*?))?(?=\s*Module\s+(?:\d+|[A-Z])|$)',
        # Numbered list format: "1.", "2.", etc.
        r'(?:^|\n)\s*(\d+)[.:\s]\s*([^\n•]*?)(?:\n([^1-9]*?))?(?=\s*(?:(?:^|\n)\s*\d+[.:\s]|$))',
        # Bullet points that might be modules
        r'(?:^|\n)\s*[•-]\s*([^•\n-]*?)(?:\n([^•-]*?))?(?=\s*(?:(?:^|\n)\s*[•-]|$))'
    ]
    
    for pattern_idx, pattern in enumerate(module_patterns):
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        if matches:
            logger.info(f"Using pattern {pattern_idx} - found {len(matches)} matches")
            
            for match in matches:
                if len(match) == 2:  # Simple format
                    module_num, module_content = match
                elif len(match) == 3:  # Format with title and content
                    module_num, module_title, module_content = match
                    if not module_content:
                        module_content = module_title
                        module_title = f"Module {module_num}"
                else:
                    continue
                
                # Clean up the content
                content = (module_content or "").strip()
                
                # Extract title from content if not already separated
                if len(match) == 2:
                    title_match = re.match(r'^([^•\n-]*?)(?:[•\n-]|$)', content)
                    module_title = title_match.group(1).strip() if title_match else f"Module {module_num}"
                
                # Clean up title and content
                clean_title = re.sub(r'^[•\-\s]*', '', module_title).strip()
                if not clean_title:
                    clean_title = f"Module {module_num}"
                
                modules.append((clean_title, content))
            
            break  # Use the first pattern that works
    
    # Enhanced fallback: try to find section headers
    if not modules:
        logger.info("No module patterns found, trying section headers")
        section_patterns = [
            r'(?:^|\n)\s*([A-Z][^•\n]*?)(?:\n([^A-Z]*?))?(?=\s*(?:(?:^|\n)\s*[A-Z]|$))',
            r'(?:^|\n)\s*([^•\n]{10,50})(?:\n([^•\n]*?))?(?=\s*(?:(?:^|\n)|$))'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if matches and len(matches) >= 2:  # At least 2 sections
                for i, match in enumerate(matches[:8]):  # Limit to 8 sections
                    title = match[0].strip() if match[0] else f"Section {i+1}"
                    content = match[1].strip() if len(match) > 1 and match[1] else title
                    if len(title) > 5 and len(content) > 10:  # Basic quality check
                        modules.append((title, content))
                break
    
    # Final fallback: split by bullets or paragraphs
    if not modules:
        logger.info("Using final fallback - splitting by structure")
        # Try bullet points
        bullet_matches = re.findall(r'[•-]\s*([^•-]*?)(?=\s*(?:[•-]|$))', text, re.DOTALL)
        if bullet_matches and len(bullet_matches) >= 3:
            for i, content in enumerate(bullet_matches[:6]):  # Limit to 6
                content = content.strip()
                if len(content) > 20:  # Substantial content
                    title_match = re.match(r'^([^.\n]{5,50})', content)
                    title = title_match.group(1).strip() if title_match else f"Point {i+1}"
                    modules.append((title, content))
        
        # If still no modules, create generic ones
        if not modules:
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            for i, para in enumerate(paragraphs[:4]):  # Max 4 paragraphs
                title = f"Section {i+1}"
                first_line = para.split('\n')[0][:50] + "..."
                modules.append((first_line, para))
    
    logger.info(f"Extracted {len(modules)} modules: {[m[0] for m in modules]}")
    return modules

def adapt_module_based_prompt(modules: List[Tuple[str, str]]) -> str:
    """
    Create a comprehensive system message based on extracted modules.
    
    Args:
        modules: List of (module_title, module_description) tuples
        
    Returns:
        Adapted system message with the module structure
    """
    if not modules:
        return None
    
    system_message = """You are a senior financial research analyst at a top-tier consulting firm. You provide comprehensive, data-driven analysis based on available context. 

Based on the user's query and the provided context, generate a detailed response structured into the following modules:

"""
    
    for i, (title, content) in enumerate(modules):
        system_message += f"\n**Module {i+1}: {title}**\n"
        
        # Extract and format bullet points or key requirements
        bullet_points = re.findall(r'(?:^|\n)\s*[-•]\s*(.*?)(?=\n\s*[-•]|\n\n|$)', content, re.DOTALL)
        
        if bullet_points:
            for bullet in bullet_points:
                cleaned_bullet = bullet.strip().replace('\n', ' ')
                if len(cleaned_bullet) > 10:  # Skip very short bullets
                    system_message += f"• {cleaned_bullet}\n"
        else:
            # Split content into logical parts
            content_parts = [part.strip() for part in content.split('\n') if part.strip()]
            for part in content_parts:
                if len(part) > 15:  # Substantial content
                    system_message += f"• {part}\n"
    
    system_message += """

**CRITICAL INSTRUCTIONS:**
- Use ONLY the information provided in the context. Do not add external knowledge.
- Each module must contain 300-400 words of substantial content.
- Include specific data points, figures, and examples from the context.
- Create tables where data allows for structured comparison.
- Cite sources for all facts and claims using exact entity names or document titles.
- Ensure every piece of provided data is utilized effectively.
- Write each sentence with 30-40+ words for comprehensive coverage.
- Format response as structured JSON matching the specified schema.

**OUTPUT FORMAT:** Respond in JSON format with each module as a separate field containing title, content, bullet_points, tables, and examples as appropriate."""

    return system_message

# === Initialize LLM ===
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

# Global variable to store current response model
current_response_model = None
llm_struct = None

def get_structured_llm(modules: List[Tuple[str, str]]):
    """Get LLM instance with structured output for the current modules."""
    global current_response_model, llm_struct
    
    # Create dynamic model based on modules
    current_response_model = create_dynamic_response_model(modules)
    
    # Create LLM with structured output
    llm_struct = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY")) \
        .with_structured_output(current_response_model)
    
    return llm_struct

# === Conversational Memory Setup ===
convo_buffers: Dict[str, ChatMessageHistory] = {}

def get_user_convo_history(session_id: str) -> ChatMessageHistory:
    if session_id not in convo_buffers:
        convo_buffers[session_id] = ChatMessageHistory()
    return convo_buffers[session_id]

# === Helper Functions ===
def dedupe_chunks(chunks: List[Any], memory_texts: Optional[Set[str]] = None) -> tuple[str, Set[str]]:
    """Deduplicate and budget context; supports posts + PDF docs + Neo4j chunks. Caps total size to ~12k chars."""
    if memory_texts is None:
        memory_texts = set()
    seen_hashes: Set[str] = set()
    context: List[str] = []
    total_budget = 20000
    per_chunk_cap = 2500
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
            source = chunk.get("source", "")
            if source == "neo4j_chunk" and text:
                text = f"[Neo4j Knowledge Graph Content] {text} [/Neo4j]"
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
    """Format PDF document content for inclusion in the prompt using enhanced metadata"""
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
                'source_url': chunk.get('source_url', ''),
                'max_score': 0.0
            }
        
        docs_by_id[doc_id]['max_score'] = max(
            docs_by_id[doc_id]['max_score'], 
            chunk.get('score', 0) + chunk.get('relevance_score', 0) * 0.3
        )
        
        content_type = "Table Data" if chunk.get('is_table', False) else "Text"
        formatted_text = chunk.get('formatted_content', chunk.get('text', ''))
        
        if formatted_text:
            docs_by_id[doc_id]['chunks'].append({
                'content_type': content_type,
                'text': formatted_text,
                'is_table': chunk.get('is_table', False)
            })
    
    sorted_docs = sorted(docs_by_id.values(), key=lambda x: x['max_score'], reverse=True)
    
    for i, doc_info in enumerate(sorted_docs):
        formatted_content += f"PDF Document {i+1}: {doc_info['title']}\n"
        formatted_content += f"Source: {doc_info['source_url']}\n"
        formatted_content += f"Relevance Score: {doc_info['max_score']:.3f}\n\n"
        
        tables = [c for c in doc_info['chunks'] if c.get('is_table', False)]
        if tables:
            formatted_content += "📊 TABLE DATA:\n"
            for table in tables[:2]:
                formatted_content += f"{table['text']}\n\n"
        
        texts = [c for c in doc_info['chunks'] if not c.get('is_table', False)]
        if texts:
            formatted_content += "📄 TEXT CONTENT:\n"
            for chunk in texts[:3]:
                formatted_content += f"{chunk['text'][:800]}...\n\n"
        
        formatted_content += "=" * 50 + "\n\n"
    
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

# === Enhanced Prompt Processing ===
def get_custom_prompt(input_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract and adapt custom prompts with flexible module structure support.
    
    Args:
        input_data: The input data that might contain a custom prompt
        
    Returns:
        Optional[str]: The custom prompt if available, None otherwise
    """
    try:
        if "prompt" in input_data:
            custom_prompt = input_data.get("prompt")
            
            if isinstance(custom_prompt, str) and len(custom_prompt.strip()) > 10:
                modules = extract_modules_from_text(custom_prompt)
                if modules:
                    return adapt_module_based_prompt(modules)
                logger.info("Using custom prompt string")
                return custom_prompt
                
            if not custom_prompt or (isinstance(custom_prompt, list) and len(custom_prompt) == 0):
                logger.info("No custom prompt available")
                return None
                
            if isinstance(custom_prompt, dict) and "content" in custom_prompt:
                content = custom_prompt["content"]
                modules = extract_modules_from_text(content)
                if modules:
                    return adapt_module_based_prompt(modules)
                logger.info("Using custom prompt from dictionary")
                return content
                
            if isinstance(custom_prompt, list) and len(custom_prompt) > 0:
                if isinstance(custom_prompt[0], dict) and "prompt" in custom_prompt[0]:
                    prompt_text = custom_prompt[0]["prompt"]
                    if prompt_text and isinstance(prompt_text, str):
                        title = custom_prompt[0].get('title', 'Untitled')
                        
                        modules = extract_modules_from_text(prompt_text)
                        if modules:
                            adapted_prompt = adapt_module_based_prompt(modules)
                            logger.info(f"Using module-based custom prompt: {title} (modules: {len(modules)})")
                            return adapted_prompt
                        
                        logger.info(f"Using custom prompt: {title}")
                        return prompt_text
                        
                elif isinstance(custom_prompt[0], str) and len(custom_prompt[0].strip()) > 10:
                    modules = extract_modules_from_text(custom_prompt[0])
                    if modules:
                        return adapt_module_based_prompt(modules)
                    logger.info("Using first prompt from list of strings")
                    return custom_prompt[0]
                    
    except Exception as e:
        logger.error(f"Error processing custom prompt: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return None

# === RAG Chain Builders ===
def build_structured_rag_chain():
    """Builds the chain for generating a structured JSON response with dynamic modules."""
    
    def get_dynamic_chain(data):
        """Create chain with dynamic model based on prompt modules."""
        custom_prompt_text = get_custom_prompt(data["input_data"])
        
        # Extract modules to create dynamic model
        modules = []
        if custom_prompt_text:
            modules = extract_modules_from_text(custom_prompt_text)

        if modules:
            print(f"Dynamic modules extracted: {[m[0] for m in modules]}")
        
        # Get structured LLM with dynamic model
        structured_llm = get_structured_llm(modules)
        
        # Create parser for current model
        parser = PydanticOutputParser(pydantic_object=current_response_model)
        format_instructions = parser.get_format_instructions()
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_message}"),
            MessagesPlaceholder("history"),
            ("human", """{question}

=== KNOWLEDGE CONTEXT ===
{chunks}

=== PDF DOCUMENT INSIGHTS ===
{pdf_content}

=== KNOWLEDGE GRAPH RELATIONSHIPS ===
{paths}

Respond in JSON as per this schema:
{format_instructions}""")
        ])
        
        # Prepare data for prompt
        prompt_data = {
            "question": data["input_data"].get("refined_query", {}).get("refined_query", "Query not found"),
            "chunks": dedupe_chunks(data["input_data"].get("qdrant_docs", []), set())[0],
            "pdf_content": format_pdf_content(data["input_data"].get("pdf_docs", [])),
            "paths": format_kg_paths(data["input_data"].get("kg_paths", [])),
            "system_message": custom_prompt_text or "You are a financial analyst. Provide detailed analysis.",
            "format_instructions": format_instructions,
            "history": []
        }
        
        # Execute the chain
        formatted_prompt = prompt_template.format_messages(**prompt_data)
        response = structured_llm.invoke(formatted_prompt)
        
        return response.dict() if hasattr(response, 'dict') else response

    return (
        {"input_data": RunnablePassthrough()}
        | RunnableLambda(get_dynamic_chain)
    )

default_convo_system_message = "You are a finance expert analyst. Answer the query in a detailed, structured format. Answer in form of a analysis report, include tables if needed. Each and every bit of data provided must be utilised. Explain everything but remain grounded. Always included references and citations for all the facts and data you provide. Try explaining the concepts in a way that is easy to understand for a non-expert. The length of report should be greater than 600 words."

def build_convo_rag_chain():
    """Builds the chain for generating a conversational text response with enhanced context."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        MessagesPlaceholder("history"),
        ("human", """{question}

=== KNOWLEDGE CONTEXT ===
{chunks}

=== PDF INSIGHTS ===
{pdf_content}

=== KNOWLEDGE GRAPH RELATIONSHIPS ===
{paths}""")
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
            "system_message": default_convo_system_message,
            "history": []
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
    input_messages_key="input_data",
    history_messages_key="history"
)

convo_chain_with_memory = RunnableWithMessageHistory(
    convo_rag_chain,
    get_session_history=get_user_convo_history,
    input_messages_key="input_data",
    history_messages_key="history"
)

def generate_structured_response(input_data: Dict[str, Any], session_id: str = "default"):
    """Generates a structured response with robust error handling and dynamic modules."""
    try:
        qdrant_count = len(input_data.get("qdrant_docs", []))
        pdf_count = len(input_data.get("pdf_docs", []))
        logger.info(f"Generating structured response with {qdrant_count} vector docs and {pdf_count} PDF docs.")

        result = structured_chain_with_memory.invoke(input_data, config={"configurable": {"session_id": session_id}})
        
        if isinstance(result, dict):
            return {"status": "success", "data": result}
        
        raise TypeError(f"Expected a dictionary from the structured chain, but got {type(result)}")

    except Exception as e:
        logger.error(f"Structured response generation failed: {e}", exc_info=True)
        return {"status": "error", "data": None, "message": f"Structured generation failed: {str(e)}"}

def generate_conversational_response(input_data: Dict[str, Any], session_id: str = "default"):
    """Generate conversational response."""
    try:
        logger.info(f"Generating conversational response for session: {session_id}")
        
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
    
    # PDF content for frontend
    pdf_content = format_pdf_content(input_data.get("pdf_docs", []))
    
    # Create combined result
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
    
    # Log errors
    if structured_response.get("status", "") == "error":
        logger.warning(f"Structured output failed: {structured_response.get('message', 'Unknown error')}")
    if convo_response.get("status", "") == "error":
        logger.warning(f"Conversational output failed: {convo_response.get('message', 'Unknown error')}")
    
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

