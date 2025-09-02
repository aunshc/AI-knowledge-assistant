from dotenv import load_dotenv
from openai import AzureOpenAI


import os
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") # This is your deployment name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME= os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

def get_openai_client():
    """Initializes and returns a single Azure OpenAI client for both embeddings and chat."""
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION]):
        print("Error: Missing Azure OpenAI environment variables. Please check your .env file.")
        return None
    try:
        client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY # Direct API key for simplicity, or use AzureKeyCredential
        )
        print("Initialized Azure OpenAI client.")
        return client
    except Exception as e:
        print(f"Error initializing Azure OpenAI client: {e}")
        return None

def get_embedding(openai_client, text):
    """Generates an embedding for the given text using the Azure OpenAI embedding model."""
    if not openai_client:
        return None
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def get_chat_completion(openai_client, messages, temperature=0.7, max_tokens=800):
    """Generates a chat completion using the Azure OpenAI chat model."""
    if not openai_client:
        return None
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting chat completion: {e}")
        return None

if __name__ == "__main__":
    client = get_openai_client()
    if client:
        # Test embedding
        test_embedding = get_embedding(client, "Hello, world!")
        if test_embedding:
            print(f"Test embedding generated (length: {len(test_embedding)}).")

        # Test chat completion
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        test_response = get_chat_completion(client, test_messages)
        if test_response:
            print(f"Test chat response: {test_response}")