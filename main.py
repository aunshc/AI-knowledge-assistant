import os
import uuid # For generating unique IDs for chunks
import shutil # For clearing the ChromaDB data directory

from openai_utils import get_openai_client, get_embedding, get_chat_completion
from chromadb_utils import get_chroma_collection, add_documents_to_chroma, query_chroma, clear_chroma_collection, CHROMA_DB_PERSIST_DIR
from document_processor import chunk_text, get_file_content # Assuming you'll adapt this for your document source

# --- Configuration ---
# Define the local directory where your documents are stored
LOCAL_DOCS_PATH = "my_local_documents" # <--- IMPORTANT: Change this to your desired folder path
os.makedirs(LOCAL_DOCS_PATH, exist_ok=True) # Ensure the directory exists

# --- Helper to determine MIME type from extension (simplified for local files) ---
def get_mime_type_from_filename(filename):
    """Simple heuristic to get MIME type based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        return "text/plain"
    elif ext == ".pdf":
        return "application/pdf"
    elif ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    # Add more as needed
    return "application/octet-stream" # Default for unknown types

def prepare_and_index_documents(chroma_collection):
    """
    Reads documents from LOCAL_DOCS_PATH, chunks them, and adds them to ChromaDB.
    """
    print(f"\n--- Preparing and Indexing Documents from '{LOCAL_DOCS_PATH}' ---")
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    # Iterate through files in the specified local directory
    for filename in os.listdir(LOCAL_DOCS_PATH):
        filepath = os.path.join(LOCAL_DOCS_PATH, filename)

        # Skip directories and non-files
        if not os.path.isfile(filepath):
            continue

        print(f"Processing '{filename}'...")

        mime_type = get_mime_type_from_filename(filename)
        content = get_file_content(filepath, mime_type)

        if content:
            chunks = chunk_text(content)
            if not chunks:
                print(f"Warning: No content or chunks extracted from '{filename}'. Skipping.")
                continue

            for i, chunk in enumerate(chunks):
                doc_id = str(uuid.uuid4()) # Generate a unique ID for each chunk
                documents_to_add.append(chunk)
                metadatas_to_add.append({"source": filename, "chunk_idx": i})
                ids_to_add.append(doc_id)
            print(f"Processed '{filename}' into {len(chunks)} chunks.")
        else:
            print(f"Could not extract content from '{filename}'. Skipping.")

    if documents_to_add:
        add_documents_to_chroma(chroma_collection, documents_to_add, metadatas_to_add, ids_to_add)
        print(f"\nSuccessfully indexed {len(documents_to_add)} document chunks.")
    else:
        print("No processable documents found in the directory to add to the knowledge base.")

def main():
    print("Welcome to the RAG System with Local ChromaDB!")

    # Initialize clients
    openai_client = get_openai_client()

    # Clear previous ChromaDB data for a fresh run
    if os.path.exists(CHROMA_DB_PERSIST_DIR):
        shutil.rmtree(CHROMA_DB_PERSIST_DIR)
        print(f"Cleaned up previous ChromaDB data in '{CHROMA_DB_PERSIST_DIR}'.")

    chroma_collection = get_chroma_collection()

    if not all([openai_client, chroma_collection]):
        print("Failed to initialize all necessary clients. Exiting.")
        return

    # Prepare and Index Documents
    prepare_and_index_documents(chroma_collection)

    print("\nKnowledge base setup complete. You can now ask questions!")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour Question: ").strip()
        if user_query.lower() == 'exit':
            print("Exiting application. Goodbye!")
            break

        if not user_query:
            print("Please enter a question.")
            continue

        # 1. Query ChromaDB for relevant documents
        print("Retrieving relevant documents from ChromaDB...")
        retrieved_chunks = query_chroma(chroma_collection, [user_query], n_results=3)

        if not retrieved_chunks:
            print("No relevant information found in the knowledge base.")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_query}
            ]
            response = get_chat_completion(openai_client, messages)
            print("\n--- Answer (from general knowledge) ---")
            print(response)
        else:
            # 2. Prepare prompt for LLM with retrieved context
            context_str = "\n\n".join([f"Content: {chunk}" for chunk in retrieved_chunks])

            system_message = (
                "You are a helpful assistant that answers questions based ONLY on the provided context. "
                "If the answer is not found in the context, state that you don't know or that the information is not available."
                "Do not make up information."
            )
            user_message = (
                f"Context:\n{context_str}\n\n"
                f"Question: {user_query}\n\n"
                "Answer:"
            )
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            # 3. Get completion from Azure OpenAI chat model
            print("Generating response with LLM...")
            response = get_chat_completion(openai_client, messages)

            print("\n--- Answer ---")
            print(response)
            print("--------------")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Create the 'my_local_documents' folder in the same directory as main.py
    # and place your .txt, .pdf, or .docx files inside it.
    # Example:
    # rag_local_chromadb/
    # ├── main.py
    # └── my_local_documents/
    #     ├── policy.txt
    #     ├── report.pdf
    #     └── faq.docx
    # -----------------

    # Ensure the local documents directory exists
    os.makedirs(LOCAL_DOCS_PATH, exist_ok=True)

    # You can optionally add some dummy files programmatically for first-time testing
    # if you don't want to create them manually:
    if not os.listdir(LOCAL_DOCS_PATH): # Only create if folder is empty
        print(f"'{LOCAL_DOCS_PATH}' is empty. Creating some dummy files for demonstration.")
        dummy_docs_content = [
            {"filename": "sample_policy.txt", "content": "Our company policy states that all employees must complete their annual compliance training by December 31st. Failure to do so may result in disciplinary action. Employees are also encouraged to participate in optional wellness programs."},
            {"filename": "sample_faq.txt", "content": "Q: How do I reset my password? A: Visit our website, click 'Forgot Password', and follow the instructions. Q: What is the return policy? A: Products can be returned within 30 days of purchase with original receipt."},
            {"filename": "sample_meeting_notes.txt", "content": "Meeting on Q3 Strategy: Key takeaways include focusing on market expansion in Europe and developing new features for the mobile app. John will lead the European expansion, and Sarah will manage mobile app development. Next meeting is October 20th."}
        ]
        for doc in dummy_docs_content:
            with open(os.path.join(LOCAL_DOCS_PATH, doc["filename"]), "w", encoding="utf-8") as f:
                f.write(doc["content"])
        print("Dummy files created. Please run the script again to index them.")
        exit() # Exit so user can see the files and rerun

    main()