from pymongo import MongoClient
from typing import Dict, Optional, Union, Any
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv
import urllib.parse
import logging
from agents.intake import process_intake, IntakeState

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('query_refiner')

# Load environment variables
load_dotenv()

class QueryRefinerAgent:
    def __init__(self, llm: ChatOpenAI, mongo_uri: Optional[str] = None):
        self.llm = llm
        self.mongo_client = MongoClient(mongo_uri) if mongo_uri else None
        
        # Use the collection object directly, not a cursor
        self.tag_db = self.mongo_client['veerive-db'] if self.mongo_client is not None else None
        
        # Access the actual collections, not cursors
        if self.tag_db is not None:
            # Using separate collections for each tag type
            self.sectors_collection = self.tag_db['sectors']
            self.countries_collection = self.tag_db['countries'] 
            self.companies_collection = self.tag_db['companies']
        else:
            self.sectors_collection = None
            self.countries_collection = None
            self.companies_collection = None

    def _verify_tag(self, tag_value: str, tag_type: str) -> Optional[str]:
        """Checks if a tag value exists in MongoDB and returns normalized name."""
        if not tag_value:
            return tag_value
            
        try:
            # Use the appropriate collection based on tag type
            if tag_type == "sector" and self.sectors_collection:
                collection = self.sectors_collection
            elif tag_type == "country" and self.countries_collection:
                collection = self.countries_collection
            elif tag_type == "company" and self.companies_collection:
                collection = self.companies_collection
            else:
                return tag_value  # No matching collection
                
            # Search for the tag in the collection
            result = collection.find_one({"name": {"$regex": f"^{tag_value}$", "$options": "i"}})
            return result["name"] if result else tag_value
            
        except Exception as e:
            logger.warning(f"Error verifying tag '{tag_value}' of type '{tag_type}': {str(e)}")
            return tag_value

    def refine(self, input_data: Union[str, Dict, IntakeState]) -> Dict:
        """
        Refine a query to extract structured tags.
        
        Args:
            input_data: Can be:
                - A string query
                - An IntakeState object from intake.py
                - A dictionary with at least a 'query' key
                
        Returns:
            Dictionary with original_query, refined_query, and tags
        """
        # Extract query and metadata from different input types
        metadata = {}
        
        if isinstance(input_data, str):
            query = input_data
        elif isinstance(input_data, dict):
            # Check if this is an IntakeState object
            if 'query' in input_data:
                query = input_data['query']
                metadata = input_data.get('metadata', {})
            else:
                query = str(input_data)
        else:
            query = str(input_data)
            
        # Include any clarifications in the prompt if they exist
        clarification_context = ""
        if isinstance(metadata, dict) and 'clarifications' in metadata:
            clarifications = metadata['clarifications']
            clarification_text = "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}"
                for item in clarifications
            ])
            clarification_context = f"\nAdditional context from clarifications:\n{clarification_text}"
            
        # Create the prompt with enriched context
        prompt = f"""
You are an intelligent query refiner. Extract the following structured tags from the user query:
- Sector (the industry or business domain)
- Country (specific geography if mentioned)
- Company (specific organization if mentioned)
- Query Type (business model, trend, impact analysis, swot, etc.)
- Subsector (like B2B, B2C, retail, wholesale, etc.)

User query: "{query}"{clarification_context}

Only reply in JSON:
{{
  "original_query": "{query}",
  "refined_query": "...",  
  "tags": {{
    "sector": "...",
    "country": "...", 
    "company": "...",
    "subsector": "...",
    "query_type": "..."
  }}
}}

Make sure the refined_query clarifies and expands the original query based on any clarifications provided.
Leave tag values empty (as "") if they are not present in the query.
"""
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up response if needed (handling markdown code blocks)
            if response_text.startswith('```'):
                parts = response_text.split('```')
                if len(parts) >= 3:
                    response_text = parts[1].strip()
                    if response_text.startswith('json'):
                        response_text = response_text[4:].strip()
            
            # Parse JSON response
            parsed = json.loads(response_text)
            
            # Verify tags against MongoDB
            tags = parsed.get("tags", {})
            for key in ["sector", "country", "company"]:
                if tags.get(key):
                    tags[key] = self._verify_tag(tags[key], key)
                    
            parsed["tags"] = tags
            
            # Include original intake metadata for traceability
            if metadata:
                parsed["metadata"] = metadata
                
            # Include request_id if available
            if isinstance(input_data, dict) and 'request_id' in input_data:
                parsed["request_id"] = input_data["request_id"]
                
            return parsed
            
        except Exception as e:
            logger.error(f"Error refining query: {str(e)}")
            return {
                "error": "LLM parsing failed", 
                "original_query": query,
                "raw_response": response_text if 'response_text' in locals() else None,
                "exception": str(e)
            }


def get_refiner():
    """Helper function to get a pre-configured QueryRefinerAgent"""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    username = "chaubeyp"
    password = urllib.parse.quote_plus("ConsTrack360")
    mongouri = f"mongodb+srv://{username}:{password}@veerive.tta8g.mongodb.net/"
    return QueryRefinerAgent(llm, mongo_uri=mongouri)


if __name__ == "__main__":
    # Example 2: Process via intake.py first
    query = input("Enter your query: ")
    
    # First process through intake agent (interactive mode)
    intake_result = process_intake(query, interactive=True)
    print("\nIntake Result:")
    print(json.dumps(intake_result, indent=2))
    
    if intake_result["status"] == "success":
        # Then pass the intake state to the refiner
        refiner = get_refiner()
        refined = refiner.refine(intake_result["data"])
        
        print("\nRefined Query Result:")
        print(json.dumps(refined, indent=2))
