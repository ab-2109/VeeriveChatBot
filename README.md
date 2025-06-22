# VeeriveChatBot

**VeeriveChatBot** is a modern full-stack question-answering chatbot built with:
- **React/Next.js** frontend (deployed on Vercel)
- **FastAPI** backend (deployed on AWS EC2 with Gunicorn + Nginx)
- **Qdrant** vector database + OpenAI embeddings
- **Cron-based batch updating** of vector store
- **Application graph structure** to track session flow & clarifications

---

## 🚀 Features

- **Conversational Q/A flow** with multi-turn support
- **Clarification** logic when questions are ambiguous
- **Vector search** on Qdrant for semantic retrieval
- **OpenAI embeddings** used for question understanding
- **Session history** tracking via unique `session_id`
- **Production-grade deployment** on AWS with HTTPS, CORS, and cron maintenance

---

## 🎨 Architecture Overview (Graph Structure)

- **User Query** --> **Intake Agent** --> **Clarification Agent** --> **Refiner Agent** --> **Retrieval Agent** --> **Generation Agent** --> **Output to Frontend**
- 1. Clone repo and install dependencies:
    ```bash
    git clone https://github.com/ab-2109/VeeriveChatBot
    cd VeeriveChatBot
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    - **Cron job definition** (`crontab -e`):
    ```
    0 2 * * * /home/ubuntu/VeeriveChatBot/run_qdrant.sh >> /home/ubuntu/qdrant_cron.log 2>&1
    ```

## File Descriptions
 - 1. **app.py**
      The main entry point for the backend application.
      Initializes the chatbot server and sets up API endpoints.
      Handles incoming user messages and coordinates responses.
      Integrates various modules such as agents and knowledge graph.
      Acts as the bridge between the frontend and backend logic.
- 2. **graph.py**
      Contains logic for building and querying the knowledge graph.
      Enables structured data retrieval to enhance chatbot responses.
      Defines relationships and entities relevant to the chatbot’s domain.
      Supports efficient graph traversal and information extraction.
      Works closely with the KnowledgeGraph module for data management.
- 3. **prompt_guidance.py**
      Retrives prompts from the MongoDB collection and stores it in Qdrant DB.
      Prompts are retrieved on the basis of top-k and similarity score.
      Feeds the prompt into the generation agent.
- 4. **qdrant_maintainer.py**
      Manages the Qdrant vector database used for semantic search.
      Handles tasks like data insertion, updating, and deletion in Qdrant.
      Ensures that the vector store stays synchronized with the knowledge base.
      Syncs the database everyday at 2 am through a cron job.
- 5. **clarification.py**
      Agent responsible for detecting and generating clarifying questions.
      Analyzes whether user input is ambiguous.
- 6. **intake.py**
      Handles preprocessing or formatting of raw user queries.
- 7. **raggen.py**
      Responsible for Retrieval-Augmented Generation (RAG) operations.
      Fetches documents from Qdrant and passes them to LLMs.
      Merges retrieval with generation for answer synthesis.
- 8. **refiner.py**
      Add metatags to the original query.
      Leverages MongoDB to add tags.
- 9. **retrieval.py**
      Handles vector-based retrieval from Qdrant.
      Encodes queries and runs nearest-neighbor search.
      Includes scoring logic, filters, and metadata processing.
      Supplies candidate documents to other agents
