import os
from openai import AzureOpenAI # Import AzureOpenAI instead of OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

# Azure OpenAI Specific Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") # This is your deployment name

def get_llm_client():
    """Initializes and returns an Azure OpenAI client."""
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME]):
        print("Error: Missing one or more Azure OpenAI environment variables. Please check your .env file.")
        return None
    try:
        # Initialize AzureOpenAI client
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )
        print(f"Initialized Azure OpenAI client with deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
        return client
    except Exception as e:
        print(f"Error initializing Azure OpenAI client: {e}")
        return None

def generate_code_and_tests(llm_client, problem_description, programming_language="python"):
    """
    Generates code and tests using the LLM based on a problem description.
    This simulates the Copilot's code generation capability.
    """
    prompt = f"""
    You are an expert software engineer. Given the following problem description from a support ticket,
    generate a code snippet in {programming_language} that fixes the described bug or implements the requested feature,
    and a corresponding unit test using Pytest.

    Problem Description (from Jira ticket):
    ---
    {problem_description}
    ---

    Please ensure the code is clean, concise, and directly addresses the problem.
    The unit test should cover the fix/feature and ideally include edge cases.

    Provide the code snippet first, enclosed in a ```{programming_language} block,
    followed by the test code, enclosed in a ```{programming_language} block.

    Example format:
    ```{programming_language}
    # Your code fix here
    ```

    ```{programming_language}
    # Your test code here (using pytest)
    ```
    """

    try:
        chat_completion = llm_client.chat.completions.create(
            # For Azure OpenAI, you use the deployment_name here
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that generates code and tests."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Adjust for creativity vs. consistency
            max_tokens=1500 # Adjust based on expected code length
        )
        response_content = chat_completion.choices[0].message.content
        print("LLM generated response.")

        # Extract code and test using regex
        code_match = re.search(r"```" + programming_language + r"\n(.*?)```", response_content, re.DOTALL)
        # Ensure the second regex search starts after the first code block to correctly find the test block
        test_match = re.search(r"```" + programming_language + r"\n(.*?)```", response_content[code_match.end():] if code_match else response_content, re.DOTALL)


        generated_code = code_match.group(1).strip() if code_match else ""
        generated_test = test_match.group(1).strip() if test_match else ""

        if not generated_code and not generated_test:
            print("Warning: Could not extract code or test from LLM response.")
            print(f"Full LLM response: {response_content}")

        return generated_code, generated_test

    except Exception as e:
        print(f"Error generating code and tests with LLM: {e}")
        return None, None

if __name__ == "__main__":
    # Example usage:
    llm_client = get_llm_client()
    if llm_client:
        sample_problem = "The 'calculate_discount' function incorrectly applies a 10% discount twice for premium users. It should only apply it once."
        code, test = generate_code_and_tests(llm_client, sample_problem)
        if code and test:
            print("\nGenerated Code:")
            print(code)
            print("\nGenerated Test:")
            print(test)