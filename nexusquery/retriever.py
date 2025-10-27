import logging
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
VECTOR_DB_PATH = Path("vector_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "gemma2"
TOP_K = 4

class RAGPipeline:
    def __init__(self):
        logging.info("Initializing RAG Pipeline...")
        self.vector_store = self._load_vector_store()
        self.llm = self._initialize_llm()
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': TOP_K})
        self.prompt_template = self._create_prompt_template()
        self.rag_chain = self._build_rag_chain()
        logging.info("RAG Pipeline initialized successfully.")

    def _load_vector_store(self):
        """Loads the ChromaDB vector store from disk."""
        if not VECTOR_DB_PATH.exists():
            raise FileNotFoundError(f"Vector database not found at {VECTOR_DB_PATH}. Please run the indexing pipeline first.")

        logging.info("Loading embedding model for vector store...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        logging.info(f"Loading vector store from {VECTOR_DB_PATH}...")
        return Chroma(persist_directory=str(VECTOR_DB_PATH), embedding_function=embeddings)

    def _initialize_llm(self):
        """Initializes the Ollama LLM."""
        logging.info(f"Initializing Ollama LLM with model: {OLLAMA_MODEL_NAME}...")
        return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0)

    def _create_prompt_template(self):
        """Creates the prompt template for the RAG chain."""
        template = """
        You are an expert assistant for the Scikit-learn documentation.
        Your task is to answer the user's question faithfully based ONLY on the following context.
        Do not make up information or use any external knowledge.
        If the context does not contain the answer, state that you cannot answer the question with the provided information.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _format_docs(self, docs):
        """Formats the retrieved documents into a single string."""
        return "\n\n".join(f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content}" for doc in docs)

    def _build_rag_chain(self):
        """Builds the LangChain RAG chain."""
        return (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str):
        """
        Queries the RAG pipeline and returns the response and source documents.
        """
        logging.info(f"Received query: {question}")

        # Get the source documents first
        source_docs = self.retriever.get_relevant_documents(question)

        # Get the answer from the RAG chain
        answer = self.rag_chain.invoke(question)

        logging.info(f"Generated answer: {answer}")

        return {
            "answer": answer,
            "sources": source_docs
        }

if __name__ == '__main__':
    # Example usage for testing
    try:
        rag_pipeline = RAGPipeline()
        test_question = "How do I handle missing data in scikit-learn?"
        result = rag_pipeline.query(test_question)

        print("--- QUESTION ---")
        print(test_question)
        print("\n--- ANSWER ---")
        print(result["answer"])
        print("\n--- SOURCES ---")
        for doc in result["sources"]:
            print(f"- {doc.metadata.get('source')}")
            # print(f"  Content: {doc.page_content[:150]}...")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
