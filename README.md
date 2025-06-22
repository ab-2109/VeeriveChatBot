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
