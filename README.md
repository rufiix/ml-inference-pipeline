# NexusQuery: Semantic Search and Q&A System for Technical Documentation

NexusQuery is a production-ready Question-Answering (Q&A) system based on the Retrieval-Augmented Generation (RAG) architecture. It allows users to ask questions in natural language to a large knowledge base (such as technical documentation) and receive precise, generated answers supported by citations from the source material.

## The Business Problem

Technical documentation for libraries like Scikit-learn, Pandas, or AWS is vast. Traditional keyword-based search (like `Ctrl+F`) often fails because it doesn't understand the semantic meaning of the query. For example, a user asking "how to handle missing data?" might not find a page that only uses the term "imputation." NexusQuery solves this by understanding the *meaning* behind the question and finding the most semantically relevant information.

## RAG Architecture

NexusQuery is built on a modern RAG architecture, which consists of two main pipelines:

### 1. Indexing Pipeline (Offline)

This pipeline is responsible for preparing the knowledge base. It's a one-time process (or run whenever the documentation is updated).

1.  **Scrape Data**: The `01_scrape_docs.py` script fetches content from the Scikit-learn documentation using `BeautifulSoup`.
2.  **Chunk Content**: The raw text is loaded and split into smaller, semantically coherent chunks using `LangChain`.
3.  **Generate Embeddings**: Each chunk is converted into a numerical vector (an embedding) using a Hugging Face sentence-transformer model (`all-MiniLM-L6-v2`).
4.  **Store in Vector DB**: These embeddings and the corresponding text chunks are stored in a `ChromaDB` vector database, which is optimized for fast similarity searches.

### 2. Query Pipeline (Online)

This pipeline is activated in real-time when a user asks a question.

1.  **User Query**: The user enters a question into the Streamlit web interface.
2.  **Generate Query Embedding**: The user's question is converted into a vector using the same embedding model.
3.  **Search Vector DB**: The system searches the ChromaDB to find the `top_k` document chunks that are most semantically similar to the query vector.
4.  **Augment Prompt**: The retrieved chunks (the "context") are combined with the original question into a carefully crafted prompt for a Large Language Model (LLM).
5.  **Generate Answer**: The prompt is sent to an LLM (such as `gemma2` running via `Ollama`), which generates an answer based *only* on the provided context. This prevents hallucinations and ensures faithfulness to the source material.
6.  **Display Results**: The generated answer and the source citations are displayed to the user.

## Tech Stack

*   **Language**: Python 3.10+
*   **Data Processing**: `BeautifulSoup`, `LangChain`
*   **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
*   **Vector Database**: `ChromaDB`
*   **RAG Framework**: `LangChain`
*   **Generative Model**: `Ollama` with `gemma2`
*   **Web Interface**: `Streamlit`
*   **Deployment**: `Docker` & `Docker Compose`

## How to Run the Project

Follow these steps to get NexusQuery up and running on your local machine.

### Prerequisites

*   Docker and Docker Compose installed.
*   An internet connection to download models and dependencies.

### Step 1: Start the Services

First, build and start the application and the Ollama services using Docker Compose.

```bash
docker-compose up -d --build
```

This command will:
- Build the Docker image for the Streamlit application.
- Start the `app` and `ollama` containers in the background.

### Step 2: Download the LLM Model

Next, you need to pull the `gemma2` model into the running Ollama container.

```bash
docker exec -it nexusquery_ollama ollama pull gemma2
```
This is a one-time setup. The model will be downloaded and stored in a Docker volume, so it will persist across container restarts.

### Step 3: Run the Indexing Pipeline

Now, run the scripts to scrape the documentation and build the vector database. These commands should be run from your host machine. Docker will execute them inside the running `app` container.

1.  **Scrape the documentation:**
    ```bash
    docker exec -it nexusquery_app python -m scripts.01_scrape_docs
    ```
    This will create a `data/scikit-learn_docs.jsonl` file.

2.  **Build the vector database:**
    ```bash
    docker exec -it nexusquery_app python -m scripts.02_build_vector_db
    ```
    This will process the data and create a persistent vector store in the `vector_db` directory.

### Step 4: Use the Application

Your NexusQuery system is now ready!

**Open your web browser and navigate to [http://localhost:8501](http://localhost:8501).**

You can now ask questions to the Scikit-learn documentation and get semantically accurate answers with source citations.

## Project Structure

```
.
├── Dockerfile
├── README.md
├── app.py
├── data/
│   └── scikit-learn_docs.jsonl
├── docker-compose.yml
├── nexusquery/
│   ├── __init__.py
│   └── retriever.py
├── requirements.txt
├── scripts/
│   ├── 01_scrape_docs.py
│   └── 02_build_vector_db.py
└── vector_db/
    └── (ChromaDB files)
```
