# Takes input from the intake agent and processes it through the clarification agent.

import os
import logging
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('clarification_agent')


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ClarificationError(Exception):
    """Exception raised for errors in the clarification process."""
    pass

def generate_clarification_question(query: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    
    try:
        if metadata and "clarifications" in metadata and metadata["clarifications"]:
            logger.info("Clarification already provided, skipping question generation")
            return {
                "clarification_needed": False,
                "clarification_questions": []
            }
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        system_prompt = """You are a helpful assistant that checks if a user's query requires clarification.
Your job is to identify if a query about business models, industries, or companies is ambiguous 
or missing important details like timeframe, region, or specific segment.

If clarification is needed, ask ONE specific question. 
Format your response as a direct question without any explanation or preamble.
If no clarification is needed, respond with "NO_CLARIFICATION_NEEDED"."""

        metadata_str = str(metadata) if metadata else "{}"
        user_prompt = f"User query: '{query}'\nAvailable metadata: {metadata_str}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        clarification_text = response.choices[0].message.content.strip()
        if clarification_text == "NO_CLARIFICATION_NEEDED":
            logger.info("No clarification needed for query")
            return {
                "clarification_needed": False,
                "clarification_questions": []
            }
        else:
            logger.info(f"Generated clarification question: {clarification_text}")
            return {
                "clarification_needed": True,
                "clarification_questions": [clarification_text]
            }
            
    except Exception as e:
        logger.error(f"Error generating clarification: {str(e)}")
        raise ClarificationError(f"Failed to generate clarification: {str(e)}")

def process_clarification_answer(query: str, answer: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    try:
        updated_metadata = dict(metadata) if metadata else {}
        
        if "clarifications" not in updated_metadata:
            updated_metadata["clarifications"] = []
        
        updated_metadata["clarifications"].append({
            "question": metadata.get("last_clarification_question", "Clarification question"),
            "answer": answer
        })
        
        if "last_clarification_question" in updated_metadata:
            del updated_metadata["last_clarification_question"]
        
        return updated_metadata
    except Exception as e:
        logger.error(f"Error processing clarification answer: {str(e)}")
        raise ClarificationError(f"Failed to process clarification answer: {str(e)}")

def clarification_node(state: Dict[str, Any], interactive_callback=None) -> Dict[str, Any]:
    """LangGraph node for handling clarifications"""
    try:
        query = state["query"]
        metadata = state["metadata"]
        
        already_clarified = (
            "clarifications" in metadata and 
            len(metadata["clarifications"]) > 0
        )
        if already_clarified:
            logger.info("Clarifications already processed, skipping")
            return state
        
        clarification_result = generate_clarification_question(query, metadata)
        
        if (clarification_result.get("clarification_needed") and 
            clarification_result.get("clarification_questions") and
            interactive_callback):
            
            question = clarification_result["clarification_questions"][0]
            logger.info(f"Interactive callback: {question}")

            return {
                **state,
                "status": "clarification_needed",
                "clarification_question": question
            }
        
        return state
        
    except Exception as e:
        logger.error(f"Clarification node error: {str(e)}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Clarification error: {str(e)}")
        return state
    
