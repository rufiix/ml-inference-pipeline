import logging
from pathlib import Path

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = Path("data") / "scikit-learn_docs.jsonl"
VECTOR_DB_PATH = Path("vector_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_documents():
    """Loads documents from the JSONL file."""
    if not DATA_PATH.exists():
        logging.error(f"Data file not found at {DATA_PATH}. Please run the scraping script first.")
        return []

    loader = JSONLoader(
        file_path=str(DATA_PATH),
        jq_schema='.',
        content_key="text",
        json_lines=True,
        metadata_func=lambda record, metadata: {"source": record.get("source_url")},
    )

    logging.info("Loading documents...")
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    """Splits documents into smaller chunks."""
    logging.info(f"Splitting {len(documents)} documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

def get_embedding_model():
    """Initializes the Hugging Face embedding model."""
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model_kwargs = {'device': 'cpu'} # Use CPU for broad compatibility
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def build_vector_db(chunks, embeddings):
    """Creates and persists the ChromaDB vector store."""
    logging.info(f"Building vector database at {VECTOR_DB_PATH}...")

    # Ensure the directory exists
    VECTOR_DB_PATH.mkdir(exist_ok=True)

    # Create the vector store. This will automatically process and store the embeddings.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_PATH)
    )

    logging.info("Vector database built successfully.")
    logging.info(f"Total documents in store: {vector_store._collection.count()}")

def main():
    """Main function to build the vector database."""
    logging.info("Starting the vector database build process.")

    documents = load_documents()
    if not documents:
        return

    chunks = split_documents(documents)

    embeddings = get_embedding_model()

    build_vector_db(chunks, embeddings)

    logging.info("Process finished.")

if __name__ == "__main__":
    main()
