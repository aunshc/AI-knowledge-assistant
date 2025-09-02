import os
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
from openai import AzureOpenAI # Import for the custom embedding function
load_dotenv()
# Define the directory where ChromaDB will store its data
CHROMA_DB_PERSIST_DIR = "./chroma_db_data"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") # This is your deployment name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")


class AzureOpenAIEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom EmbeddingFunction for ChromaDB using Azure OpenAI."""
    def __init__(self, api_key, azure_endpoint, api_version, deployment_name):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self.deployment_name = deployment_name

    def __call__(self, texts):
        # The texts argument is expected to be a list of strings
        # Azure OpenAI embedding API expects a list of strings
        response = self.client.embeddings.create(
            input=texts,
            model=self.deployment_name
        )
        # Extract the embedding vectors from the response
        return [data.embedding for data in response.data]

def get_chroma_collection(collection_name="rag_documents"):
    """
    Initializes and returns a ChromaDB client and collection.
    """
    try:
        # Initialize ChromaDB client (persistent to save data to disk)
        client = chromadb.PersistentClient(path=CHROMA_DB_PERSIST_DIR)

        # Initialize Azure OpenAI Embedding Function for ChromaDB
        if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME]):
            print("Error: Missing Azure OpenAI embedding environment variables. Cannot create embedding function.")
            return None

        embedding_function = AzureOpenAIEmbeddingFunction(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        )

        # Get or create the collection
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function # Pass the custom embedding function here
        )
        print(f"ChromaDB collection '{collection_name}' initialized. Data stored in {CHROMA_DB_PERSIST_DIR}")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB or embedding function: {e}")
        return None

def add_documents_to_chroma(collection, documents, metadatas=None, ids=None):
    """Adds documents (chunks) to the ChromaDB collection."""
    if not collection:
        return False
    try:
        # ChromaDB expects lists of texts, metadatas, and ids
        # Embeddings are generated automatically by the embedding_function passed during collection creation
        collection.add(
            documents=documents,
            metadatas=metadatas if metadatas else [{}] * len(documents),
            ids=ids if ids else [f"doc_{i}" for i in range(len(documents))]
        )
        print(f"Added {len(documents)} documents to the ChromaDB collection.")
        return True
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")
        return False

def query_chroma(collection, query_texts, n_results=5):
    """Queries the ChromaDB collection for relevant documents."""
    if not collection:
        return []
    try:
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results
        )
        # results['documents'] is a list of lists. We want the first (and usually only) inner list.
        retrieved_docs = results['documents'][0] if results and results['documents'] else []
        print(f"Queried ChromaDB for '{query_texts[0]}' and found {len(retrieved_docs)} results.")
        return retrieved_docs
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

def clear_chroma_collection(collection_name="rag_documents"):
    """Deletes and recreates the ChromaDB collection to clear its contents."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PERSIST_DIR)
        client.delete_collection(name=collection_name)
        print(f"ChromaDB collection '{collection_name}' cleared.")
        return True
    except Exception as e:
        print(f"Error clearing ChromaDB collection: {e}")
        return False

if __name__ == "__main__":
    # Example usage:
    # Ensure your .env has Azure OpenAI embedding details

    # Clear previous data for a fresh test
    clear_chroma_collection()

    chroma_collection = get_chroma_collection()
    if chroma_collection:
        sample_docs = []
        sample_metadatas = []
        sample_ids = [f"id_{i}" for i in range(len(sample_docs))]

        add_documents_to_chroma(chroma_collection, sample_docs, sample_metadatas, sample_ids)

        query = "Who is the Product Manager for Infrastructure Monitoring?"
        results = query_chroma(chroma_collection, [query], n_results=2)
        if results:
            print(f"\nQuery: '{query}'")
            for i, doc in enumerate(results):
                print(f"  Result {i+1}: {doc}")