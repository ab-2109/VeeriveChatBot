import os
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import langchain.callbacks
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv
import re
import json

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
    content: str = Field(..., description="Module content - must be 300-400 words, comprehensive and detailed")
    bullet_points: List[str] = Field(default_factory=list, description="Key points")
    tables: List[TableData] = Field(default_factory=list, description="Tables if any")
    examples: List[str] = Field(default_factory=list, description="Examples if any")

# Global variables for models
current_response_model = None
llm_struct = None
USE_LLM_MODULE_PARSE = os.getenv("ENABLE_LLM_MODULE_PARSE", "0") == "1"

# === Advanced Module Parsing Functions ===
def parse_modules_robust(text: str) -> List[Dict[str, Any]]:
    """
    Robust module parsing that handles various formats and inline modules.
    Returns list of modules with sequential numbering for schema creation.
    """
    stripped = (text or "").strip()
    if not stripped:
        return []

    # Strip surrounding triple quotes
    if (stripped.startswith('"""') and stripped.endswith('"""')) or \
       (stripped.startswith("'''") and stripped.endswith("'''")):
        stripped = stripped[3:-3].strip()

    logger.info(f"Parsing text length: {len(stripped)}")
    logger.info(f"First 200 chars: {stripped[:200]}")
    
    modules = []
    
    # Method 1: Split-based parsing for inline modules
    # This handles cases where modules are on the same line or poorly separated
    module_pattern = r'Module\s+(\d+)\s*[—\-–—]\s*'
    splits = re.split(module_pattern, stripped, flags=re.IGNORECASE)
    
    if len(splits) > 2:  # We have actual splits
        logger.info(f"Found {(len(splits)-1)//2} modules using split method")
        
        # Process splits: [preamble, num1, content1, num2, content2, ...]
        for i in range(1, len(splits), 2):
            if i + 1 < len(splits):
                mod_num = int(splits[i])
                content = splits[i + 1].strip()
                
                # Extract title from content (first meaningful part before bullet or long description)
                title_patterns = [
                    r'^([^•\n]{10,80}?)(?:\s*[•\n]|$)',  # Before bullet or newline
                    r'^([^•]{10,80}?)(?:\s*[•]|$)',      # Before bullet
                    r'^(.{10,80}?)(?:\s|$)'              # First 10-80 chars
                ]
                
                title = f"Module {mod_num}"
                for pattern in title_patterns:
                    title_match = re.match(pattern, content)
                    if title_match:
                        candidate = title_match.group(1).strip()
                        # Clean up common artifacts
                        candidate = re.sub(r'\s*Module\s+\d+.*$', '', candidate, flags=re.IGNORECASE)
                        candidate = re.sub(r'[•\-—]+.*$', '', candidate)
                        if len(candidate) > 5 and len(candidate) < 100:
                            title = candidate
                            break
                
                modules.append({
                    "module": len(modules) + 1,  # Sequential numbering for schema
                    "original_number": mod_num,
                    "title": title,
                    "content": content
                })
                logger.info(f"Parsed Module {mod_num} -> Schema Module {len(modules)}: {title[:50]}...")
    
    # Method 2: Regex-based parsing as fallback
    if not modules:
        logger.info("Split method failed, trying regex method")
        module_regex = r'Module\s+(\d+)\s*[—\-–—]\s*([^•]*?)(?=Module\s+\d+|$)'
        matches = list(re.finditer(module_regex, stripped, re.IGNORECASE | re.DOTALL))
        
        for i, match in enumerate(matches):
            mod_num = int(match.group(1))
            full_content = match.group(2).strip()
            
            # Extract title (first line or first part before bullet)
            title_match = re.match(r'^([^•\n]{5,80}?)(?:[•\n]|$)', full_content)
            title = title_match.group(1).strip() if title_match else f"Module {mod_num}"
            
            modules.append({
                "module": i + 1,  # Sequential numbering
                "original_number": mod_num,
                "title": title,
                "content": full_content
            })
            logger.info(f"Regex parsed Module {mod_num} -> Schema Module {i+1}: {title[:50]}...")
    
    # Method 3: Bullet-point based parsing if still no modules
    if not modules:
        logger.info("Regex method failed, trying bullet-point extraction")
        modules = _extract_from_bullets(stripped)
    
    # Final validation and cleanup
    if not modules:
        logger.warning("All parsing methods failed, creating default modules")
        modules = [
            {"module": 1, "title": "Analysis Overview", "content": stripped[:500] + "..."},
            {"module": 2, "title": "Key Insights", "content": stripped[500:1000] + "..."},
            {"module": 3, "title": "Strategic Implications", "content": stripped[1000:] or "Additional analysis needed"}
        ]
    
    logger.info(f"Final module count: {len(modules)}")
    return modules

def _extract_from_bullets(text: str) -> List[Dict[str, Any]]:
    """Extract modules from bullet-pointed or structured text."""
    modules = []
    
    # Try different bullet patterns
    bullet_patterns = [
        r'[•▪▫–—-]\s*([^•▪▫–—-\n]{20,}?)(?=\s*[•▪▫–—-]|$)',
        r'(?:^|\n)\s*[-*]\s*([^-*\n]{20,}?)(?=\s*(?:^|\n)\s*[-*]|$)',
        r'(?:^|\n)\s*(\d+\.?\s*[^0-9\n]{20,}?)(?=\s*(?:^|\n)\s*\d+\.|$)'
    ]
    
    for pattern in bullet_patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if matches and len(matches) >= 2:
            for i, content in enumerate(matches[:8]):  # Max 8 modules
                content = content.strip()
                if len(content) > 15:
                    # Extract title from first part
                    title_match = re.match(r'^([^.!?]{5,60})', content)
                    title = title_match.group(1).strip() if title_match else f"Section {i+1}"
                    
                    modules.append({
                        "module": i + 1,
                        "title": title,
                        "content": content
                    })
            break
    
    return modules

# === Model Creation ===
def create_dynamic_response_model(modules: List[Dict[str, Any]]) -> type:
    """Create dynamic Pydantic model with module_N fields based on parsed modules."""
    fields = {}
    
    # Ensure we have at least some modules
    if not modules:
        modules = [
            {"title": "Executive Analysis", "content": "General strategic analysis"},
            {"title": "Key Metrics", "content": "Important quantitative indicators"},
            {"title": "Strategic Recommendations", "content": "Strategic opportunities and next steps"}
        ]
    
    # Create fields for each module
    for i, module in enumerate(modules, 1):
        field_name = f"module_{i}"
        title = module.get("title", f"Module {i}")
        content_desc = module.get("content", "")[:200] + "..." if len(module.get("content", "")) > 200 else module.get("content", "")
        
        # Create comprehensive field description
        field_desc = f"MANDATORY: Complete analysis for '{title}'. Content requirements: {content_desc}. Must provide 300-400 words with comprehensive coverage including all specified elements, metrics, examples, and detailed insights."
        
        fields[field_name] = (
            DynamicModuleData, 
            Field(..., description=field_desc)
        )
    
    logger.info(f"Created dynamic model with {len(fields)} REQUIRED modules: {list(fields.keys())}")
    
    return create_model('DynamicRAGResponse', **fields, __base__=BaseModel)

def get_structured_llm(modules: List[Dict[str, Any]]):
    """Get LLM instance with structured output for the current modules."""
    global current_response_model, llm_struct
    
    # Create dynamic model based on modules
    current_response_model = create_dynamic_response_model(modules)
    
    # Use GPT-4 with function calling for structured output (more reliable)
    llm_struct = ChatOpenAI(
        model="gpt-4.1",
        temperature=1,
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_structured_output(current_response_model, method="function_calling")
    
    return llm_struct

# === Prompt Processing Utilities ===
def coerce_prompt_to_text(custom_prompt) -> str:
    """Coerce arbitrary prompt container into a plain text string."""
    if isinstance(custom_prompt, str):
        return custom_prompt
    if isinstance(custom_prompt, dict):
        return (custom_prompt.get("content") or custom_prompt.get("prompt") or "")
    if isinstance(custom_prompt, list):
        parts = []
        for item in custom_prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("content") or item.get("prompt") or "")
        return "\n\n".join(p for p in parts if p)
    return ""

def format_modules_for_prompt(modules: List[Dict[str, Any]]) -> str:
    """Format modules into a clear prompt structure."""
    formatted_sections = []
    
    for module in modules:
        section = f"[Module {module['module']} — {module['title']}]\n{module['content'].strip()}"
        formatted_sections.append(section)
    
    return "\n\n".join(formatted_sections)

def get_custom_prompt(input_data: Dict[str, Any]) -> Optional[str]:
    """Build comprehensive system prompt with module requirements."""
    try:
        raw_prompt = coerce_prompt_to_text(input_data.get("prompt"))
        if not raw_prompt or len(raw_prompt.strip()) < 5:
            return None

        modules = parse_modules_robust(raw_prompt)
        formatted_modules = format_modules_for_prompt(modules)

        # Comprehensive system instructions
        critical_instructions = f"""CRITICAL EXECUTION REQUIREMENTS:

1. MANDATORY MODULE COMPLETION:
   - You MUST generate content for ALL {len(modules)} modules specified above
   - Each module is REQUIRED and must contain substantial, detailed analysis
   - NO module may be left empty, incomplete, or with placeholder content

2. CONTENT QUALITY STANDARDS:
   - Each module must contain 300-400 words of comprehensive, data-rich content
   - Include specific examples, metrics, and detailed insights for each module
   - Address ALL requirements explicitly mentioned in each module description
   - If module mentions comparisons, scenarios, personas, or metrics - include ALL of them

3. STRUCTURE REQUIREMENTS:
   - Use tables for comparative data where specified
   - Include bullet points for key insights within each module
   - Provide specific examples and concrete details, not generalities
   - Ensure each module is substantive and actionable

4. JSON OUTPUT REQUIREMENTS:
   - Return ONLY valid JSON with no markdown formatting
   - Every module field (module_1, module_2, etc.) must be populated
   - Each module must have title, content, and relevant supporting elements

FAILURE TO COMPLETE ALL MODULES WILL RESULT IN AN INCOMPLETE ANALYSIS."""

        return f"{formatted_modules}\n\n{critical_instructions}"
        
    except Exception as e:
        logger.error(f"Error processing custom prompt: {e}")
        return None

# === Conversation Memory Setup ===
convo_buffers: Dict[str, ChatMessageHistory] = {}

def get_user_convo_history(session_id: str) -> ChatMessageHistory:
    if session_id not in convo_buffers:
        convo_buffers[session_id] = ChatMessageHistory()
    return convo_buffers[session_id]

# === Helper Functions ===
def format_chunks(chunks: List[Any]) -> str:
    """Format chunks without deduplication."""
    context = []
    
    for chunk in chunks or []:
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
        
        if text:
            context.append(text)
    
    return "\n\n---\n\n".join(context)

def format_pdf_content(pdf_docs: List[Dict[str, Any]]) -> str:
    """Format PDF document content for inclusion in the prompt."""
    if not pdf_docs:
        return ""
    
    formatted_content = "\n\n=== RELEVANT PDF DOCUMENTS ===\n\n"
    
    for chunk in pdf_docs:
        formatted_content += f"{chunk.get('text', '')[:800]}...\n\n"
        formatted_content += "=" * 50 + "\n\n"
    
    return formatted_content

def format_kg_paths(paths: List[Dict[str, Any]]) -> str:
    """Format knowledge graph paths for prompt inclusion."""
    output = []
    for p in paths:
        if isinstance(p, dict):
            if "path" in p:
                nodes = " → ".join(str(n.get("id", n)) for n in p["path"] if isinstance(n, dict))
                output.append(f"- {nodes}")
            elif "title" in p:
                output.append(f"- [{p['title']}]({p.get('url', '')})")
    return "\n".join(output)

# === RAG Chain Builder ===
def build_structured_rag_chain():
    """Build chain that dynamically creates schema based on parsed modules."""
    
    def get_dynamic_chain(data):
        input_data = data["input_data"]

        raw_prompt = coerce_prompt_to_text(input_data.get("prompt"))
        if not raw_prompt.strip():
            raw_prompt = "You are a financial analyst. Provide comprehensive structured analysis with multiple detailed modules covering all aspects of the topic."

        # Parse modules and create dynamic schema
        modules = parse_modules_robust(raw_prompt)
        logger.info(f"Building chain with {len(modules)} modules: {[m['title'][:50] for m in modules]}")

        structured_llm = get_structured_llm(modules)
        parser = PydanticOutputParser(pydantic_object=current_response_model)
        format_instructions = parser.get_format_instructions()

        system_message = get_custom_prompt({"prompt": raw_prompt}) or raw_prompt

        # Enhanced prompt template with strong module requirements
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

🚨 CRITICAL REQUIREMENT 🚨
You MUST provide comprehensive analysis for ALL modules specified in the system message.
- Each module requires 300-400 words of detailed, substantive content
- Address every requirement mentioned in each module description
- Include specific examples, metrics, comparisons, and detailed insights
- ALL module fields in your JSON response are mandatory and must be fully populated
- Incomplete responses will be rejected

Generate your response in the following JSON format:
{format_instructions}""")
        ])

        prompt_data = {
            "question": input_data.get("refined_query", {}).get("refined_query", "Provide comprehensive analysis"),
            "chunks": format_chunks(input_data.get("qdrant_docs", [])),
            "pdf_content": format_pdf_content(input_data.get("pdf_docs", [])),
            "paths": format_kg_paths(input_data.get("kg_paths", [])),
            "system_message": system_message,
            "format_instructions": format_instructions,
            "history": []
        }

        formatted_messages = prompt_template.format_messages(**prompt_data)
        logger.info(f"Invoking LLM with {len(modules)} required modules...")
        
        try:
            response = structured_llm.invoke(formatted_messages)
            
            # Validate response completeness
            if hasattr(response, "dict"):
                response_dict = response.dict()
                expected_modules = [f"module_{i}" for i in range(1, len(modules) + 1)]
                populated_modules = [k for k in response_dict.keys() if k.startswith("module_")]
                
                logger.info(f"Expected: {len(expected_modules)} modules, Got: {len(populated_modules)} modules")
                
                if len(populated_modules) < len(expected_modules):
                    logger.warning(f"Incomplete response: {len(populated_modules)}/{len(expected_modules)} modules populated")
                    # Log which modules are missing
                    missing = set(expected_modules) - set(populated_modules)
                    logger.warning(f"Missing modules: {missing}")
                else:
                    logger.info("✅ All modules successfully populated")
                
                return response_dict
            else:
                return response
                
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    return {"input_data": RunnablePassthrough()} | RunnableLambda(get_dynamic_chain)

# === Response Generation ===
structured_rag_chain = build_structured_rag_chain()

structured_chain_with_memory = RunnableWithMessageHistory(
    structured_rag_chain,
    get_session_history=get_user_convo_history,
    input_messages_key="input_data",
    history_messages_key="history"
)

def generate_structured_response(input_data: Dict[str, Any], session_id: str = "default"):
    """Generate structured response with comprehensive error handling."""
    try:
        qdrant_count = len(input_data.get("qdrant_docs", []))
        pdf_count = len(input_data.get("pdf_docs", []))
        logger.info(f"Generating structured response with {qdrant_count} vector docs and {pdf_count} PDF docs")

        result = structured_chain_with_memory.invoke(
            input_data, 
            config={"configurable": {"session_id": session_id}}
        )
        
        if isinstance(result, dict):
            # Log success metrics
            module_count = len([k for k in result.keys() if k.startswith("module_")])
            logger.info(f"✅ Successfully generated response with {module_count} modules")
            return {"status": "success", "data": result}
        
        raise TypeError(f"Expected dict from structured chain, got {type(result)}")

    except Exception as e:
        logger.error(f"Structured response generation failed: {e}", exc_info=True)
        return {
            "status": "error", 
            "data": None, 
            "message": f"Structured generation failed: {str(e)}"
        }

def run_rag_generator(input_data: Dict[str, Any], session_id: str = "default"):
    """Main RAG generation function with comprehensive logging."""
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a string")
    
    logger.info(f"🚀 Starting RAG generation for session: {session_id}")
    
    # Log comprehensive input statistics
    query = input_data.get("refined_query", {}).get("refined_query", "Unknown query")
    qdrant_count = len(input_data.get("qdrant_docs", []))
    pdf_count = len(input_data.get("pdf_docs", []))
    kg_paths_count = len(input_data.get("kg_paths", []))
    kg_insights_count = len(input_data.get("kg_insights", []))
    
    logger.info(f"Query: {query[:100]}...")
    logger.info(f"📊 Resources: {qdrant_count} vector docs, {pdf_count} PDFs, {kg_paths_count} KG paths, {kg_insights_count} KG insights")
    
    # Generate structured response
    structured_response = generate_structured_response(input_data, session_id)
    pdf_content = format_pdf_content(input_data.get("pdf_docs", []))
    
    # Compile final result
    result = {
        "structured": {
            "status": structured_response.get("status", "error"),
            "data": structured_response.get("data", {}),
            "error": structured_response.get("message", None) if structured_response.get("status", "") == "error" else None
        },
        "pdf_content": pdf_content
    }
    
    # Validate result completeness
    if structured_response.get("status", "") == "error" and not pdf_content:
        logger.error("❌ Both structured output and PDF content failed")
        raise Exception("RAG generation failed: no usable output generated")
    
    # Final logging
    status = result['structured']['status']
    pdf_length = len(result['pdf_content'])
    if status == "success":
        module_count = len([k for k in result['structured']['data'].keys() if k.startswith('module_')])
        logger.info(f"✅ RAG generation completed successfully - {module_count} modules, {pdf_length} chars PDF content")
    else:
        logger.warning(f"⚠️ RAG generation completed with errors - status: {status}, PDF content: {pdf_length} chars")
    
    return result

# === Testing and Debug Utilities ===
def test_prompt_pipeline(
    query: str,
    custom_prompt: Optional[str] = None,
    qdrant_docs: Optional[List[Dict[str, Any]]] = None,
    pdf_docs: Optional[List[Dict[str, Any]]] = None,
    kg_paths: Optional[List[Dict[str, Any]]] = None,
    kg_insights: Optional[List[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Test and debug the prompt pipeline without making LLM calls."""
    base_input = {
        "refined_query": {"refined_query": query},
        "prompt": custom_prompt,
        "qdrant_docs": qdrant_docs or [],
        "pdf_docs": pdf_docs or [],
        "kg_paths": kg_paths or [],
        "kg_insights": kg_insights or []
    }
    if extra:
        base_input.update(extra)

    # Test module parsing
    modules = parse_modules_robust(coerce_prompt_to_text(custom_prompt))
    system_message = get_custom_prompt(base_input) or "Default system message"
    
    # Format context
    context_text = format_chunks(base_input["qdrant_docs"])
    pdf_text = format_pdf_content(base_input["pdf_docs"])
    kg_path_text = format_kg_paths(base_input["kg_paths"])

    return {
        "query": query,
        "modules_found": len(modules),
        "modules_summary": [{"module": m["module"], "title": m["title"][:50], "content_length": len(m["content"])} for m in modules],
        "system_message_length": len(system_message),
        "context_length": len(context_text),
        "pdf_content_length": len(pdf_text),
        "kg_paths_length": len(kg_path_text),
        "system_message_preview": system_message[:500] + "..." if len(system_message) > 500 else system_message
    }

# === Test Entry Point ===
if __name__ == "__main__":
    # Test with the sample BNPL prompt
    sample_prompt = """Module 1 — Executive Summary

Summarizes why BNPL is strategically relevant in the sector and geography.

Highlights key market shifts, pain points, and strategic opportunities.

Includes 2—3 headline contrasts between BNPL and traditional credit models.

Module 2 — KPI Enhancements (Make It Measurable)

Outlines core BNPL metrics: GMV, ticket size, CAC, repeat rate, delinquency.

Includes sector-specific reference metrics (e.g., % financed transactions, OOP burden, claim cycles).

Establishes a baseline to track adoption, performance, and unit economics.

Module 3 — Market Segmentation & Demand Drivers

Segments BNPL usage by product/service type, user tier, and geography.

Highlights key behavioral and financial drivers influencing adoption.

Surfaces pockets of demand variation (e.g., discretionary vs urgent use cases).

Module 4 — Competitive Landscape Mapping

Categorizes players by operating model: direct lender, aggregator, platform enabler.

Compares differentiation: underwriting IP, distribution, sector focus, embedded design.

Includes monetization strategies and a structured mapping table.

Module 5 — Regulatory & Risk Landscape

Summarizes applicable digital lending, KYC, and bureau compliance requirements.

Flags sector-specific regulatory nuances and emerging governance models.

Highlights key risks: fraud exposure, mis-selling, overextension, regulatory friction.

Module 6 — Consumer Persona Deep Dive

Presents 2—3 representative user personas contextualized to the sector.

Captures financial context, usage motivation, repayment behavior, and triggers.

Illuminates real-life friction points and adoption pathways.

Module 7 — Embedded Finance & UX Integration

Maps where and how BNPL is embedded in the user journey (checkout, PoS, billing, app).

Shows how design influences activation, drop-offs, and overall conversion.

Highlights embedded models that deliver frictionless access at point-of-need.

Module 8 — Strategic Opportunities & White Spaces

Identifies underserved categories, segments, and locations.

Highlights adjacent use cases: recurring payments, co-pay financing, deferred insurance.

Surfaces partnership opportunities (e.g., insurers, employers, aggregators).

Module 9 — Global Benchmarks & Learnings

Showcases 2—3 relevant international BNPL models from similar verticals.

Distills success factors: CAC control, embedded UX, risk mitigation, hybrid models.

Highlights localization needs and regulatory contrasts.

Module 10 — Future Outlook Scenarios

Scenario A (Optimistic): High ecosystem alignment, CAC efficiencies, broad merchant uptake.

Scenario B (Moderate): Gradual growth, localized adoption, regulatory caution.

Scenario C (Risky): Overregulation, credit misuse, or institutional resistance.
"""
    
    print("=" * 60)
    print("TESTING MODULE PARSING")
    print("=" * 60)
    
    # Test module parsing directly
    modules = parse_modules_robust(sample_prompt)
    print(f"Parsed {len(modules)} modules:")
    for module in modules:
        print(f"  Module {module['module']}: {module['title']}")
    
    print("\n" + "=" * 60)
    print("TESTING FULL PIPELINE")
    print("=" * 60)
    
    # Test full pipeline
    debug_result = test_prompt_pipeline(
        query="Provide comprehensive BNPL strategic analysis",
        custom_prompt=sample_prompt
    )
    
    print(json.dumps(debug_result, indent=2))
    
    print(f"\n🎯 RESULT: Found {debug_result['modules_found']} modules for schema generation")