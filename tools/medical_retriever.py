# tools/medical_retriever.py (Updated for Local Mac Usage)
import os
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# This path MUST match the name of the folder you unzipped from Colab
VECTOR_STORE_PATH = "vector_store/faiss_pubmed_full_large_model"

class StaticRetriever:
    """
    Loads the pre-built, static FAISS vector store for medical knowledge retrieval.
    """
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self._retriever = None

    def _load_vector_store(self):
        if not os.path.exists(VECTOR_STORE_PATH):
            raise FileNotFoundError(
                f"Vector store not found at '{VECTOR_STORE_PATH}'. "
                "Ensure you have downloaded and unzipped the index from Colab."
            )
        print(f"-> Loading pre-built vector store from '{VECTOR_STORE_PATH}'...")
        self.vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True
        )
        self._retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print("-> Vector store loaded successfully.")

    def get_retriever(self) -> VectorStoreRetriever:
        """Public method to get the initialized retriever."""
        if not self._retriever:
            self._load_vector_store()
        return self._retriever

# --- Self-Testing Block (Updated for Local Mac) ---
if __name__ == '__main__':
    

    print("--- Testing StaticRetriever on Local Mac ---")

    # This MUST match the model used in Colab for the index to work.
    print("-> Initializing local embeddings model to run on Mac's MPS...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'}, # Use 'mps' for Apple Silicon
        encode_kwargs={'normalize_embeddings': True}
    )
    
    static_retriever_tool = StaticRetriever(embeddings=embeddings)
    retriever = static_retriever_tool.get_retriever()
    
    test_query = "What is the connection between GERD and non-cardiac chest pain?"
    print(f"\n-> Testing with query: '{test_query}'")
    search_results = retriever.invoke(test_query)
    
    print(f"\n--- Retriever Test Results ---")
    for doc in search_results:
        print(f"\n[Source: {doc.metadata.get('source')}]")
        print(doc.page_content[:400] + "...") # Print snippet
        print("-" * 50)