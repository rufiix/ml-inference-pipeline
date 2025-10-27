import streamlit as st
from nexusquery.retriever import RAGPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration ---
st.set_page_config(
    page_title="NexusQuery",
    page_icon="🔎",
    layout="wide"
)

# --- Title and Description ---
st.title("NexusQuery: Zapytaj dokumentację Scikit-learn 🔎")
st.markdown("""
Witamy w NexusQuery! Ta aplikacja umożliwia zadawanie pytań w języku naturalnym do dokumentacji technicznej Scikit-learn.
System wykorzystuje architekturę **Retrieval-Augmented Generation (RAG)**, aby znaleźć najbardziej trafne fragmenty
i na ich podstawie wygenerować precyzyjną odpowiedź.
""")

# --- RAG Pipeline Initialization ---
@st.cache_resource
def load_rag_pipeline():
    """
    Loads the RAGPipeline instance and caches it for performance.
    """
    try:
        pipeline = RAGPipeline()
        return pipeline
    except FileNotFoundError:
        st.error(
            "Baza wektorowa nie została znaleziona. "
            "Upewnij się, że uruchomiłeś pipeline indeksujący (`scripts/02_build_vector_db.py`) przed uruchomieniem aplikacji."
        )
        return None
    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd podczas inicjalizacji pipeline'u RAG: {e}")
        logging.error(f"RAG Pipeline initialization failed: {e}")
        return None

rag_pipeline = load_rag_pipeline()

# --- Main Application Logic ---
if rag_pipeline:
    # --- User Input ---
    question = st.text_input(
        "Zadaj pytanie dotyczące dokumentacji Scikit-learn:",
        placeholder="np. Jak radzić sobie z brakującymi danymi?",
        help="Wpisz swoje pytanie i naciśnij Enter lub kliknij przycisk 'Zapytaj'."
    )

    if st.button("Zapytaj 🗣️", type="primary") and question:
        st.markdown("---")
        # --- Processing and Output ---
        with st.spinner("Przetwarzanie zapytania... Proszę czekać..."):
            try:
                result = rag_pipeline.query(question)
                answer = result['answer']
                sources = result['sources']

                # Display the answer
                st.subheader("Odpowiedź:")
                st.markdown(answer)
                st.markdown("---")

                # Display the sources
                st.subheader("Źródła:")
                if sources:
                    for i, doc in enumerate(sources):
                        with st.expander(f"Źródło {i+1}: {doc.metadata.get('source', 'N/A')}"):
                            st.markdown(doc.page_content)
                else:
                    st.warning("Nie znaleziono żadnych źródeł dla tego zapytania.")

            except Exception as e:
                st.error(f"Wystąpił błąd podczas przetwarzania Twojego zapytania: {e}")
                logging.error(f"Error during query processing: {e}")

    elif not question and st.button("Zapytaj 🗣️", type="primary"):
        st.warning("Proszę wpisać pytanie.")
else:
    st.warning("Aplikacja nie może działać, dopóki problem z pipeline'em RAG nie zostanie rozwiązany.")

# --- Footer ---
st.markdown("---")
st.markdown("Stworzone z ❤️ przez Agenta 'Vector'")
