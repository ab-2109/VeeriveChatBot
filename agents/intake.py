import uuid
import logging
from typing import TypedDict, Optional, Dict, Any, Union
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('intake_agent')

class IntakeState(TypedDict):
    query: str
    user_id: Optional[str]
    timestamp: str
    request_id: str
    metadata: Dict[str, Any]

class InvalidInputError(Exception):
    pass

def validate_query(query: str) -> bool:
    if not query or not query.strip():
        return False
    if len(query.strip()) < 3:
        return False
    return True

def update_metadata_with_tag(metadata: dict, tag_value: str) -> dict:
    if "tags" not in metadata:
        metadata["tags"] = []
    if tag_value:
        metadata["tags"].append(tag_value.strip())
    return metadata

def intake_agent(input_data: Union[str, dict], interactive: bool = False) -> IntakeState:
    try:
        # Handle case where input_data is just the query string
        if isinstance(input_data, str):
            query = input_data.strip()
            input_data = {
                'query': query,
                'user_id': None,
                'metadata': {},
            }
        else:
            query = input_data.get('query', '').strip()
            
        if not validate_query(query):
            raise InvalidInputError("Query is required and must not be empty or too short")

        user_id = input_data.get('user_id')
        client_metadata = input_data.get('metadata', {})

        now = datetime.now()
        request_id = str(uuid.uuid4())

        metadata = {
            'source': input_data.get('source', 'console' if interactive else 'api'),
            'client_ip': input_data.get('client_ip'),
            'user_agent': input_data.get('user_agent'),
            'session_id': input_data.get('session_id'),
            **client_metadata
        }

        intake_state = {
            'query': query,
            'user_id': user_id,
            'timestamp': now.isoformat(),
            'request_id': request_id,
            'metadata': metadata
        }

        logger.info(f"Processed intake for request_id={request_id}, user_id={user_id}")
        return intake_state

    except InvalidInputError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in intake agent: {str(e)}")
        raise RuntimeError(f"Intake processing failed: {str(e)}")

def process_intake(input_data: Union[str, dict], interactive: bool = False) -> Dict[str, Any]:
    try:
        return {
            "status": "success",
            "data": intake_agent(input_data, interactive)
        }
    except InvalidInputError as e:
        return {
            "status": "error",
            "error_type": "invalid_input",
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"Error in process_intake: {str(e)}")
        return {
            "status": "error",
            "error_type": "system_error",
            "message": "An unexpected error occurred"
        }
    
if __name__ == "__main__":
    # Example usage with just a query string as input
    query = "What are the dominant business models in BNPL?"
    result = process_intake(query, interactive=False)
    print("\nFinal result:")
    print(result)