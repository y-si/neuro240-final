import os
import openai
from anthropic import Anthropic
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Key Configuration
# These will be loaded from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
LLAMA_API_ENDPOINT = os.getenv("LLAMA_API_ENDPOINT")

# Verify API keys are set
missing_keys = []
if not LLAMA_API_KEY:
    missing_keys.append("LLAMA_API_KEY")
if not LLAMA_API_ENDPOINT:
    missing_keys.append("LLAMA_API_ENDPOINT")

if missing_keys:
    raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

# Initialize OpenAI client if key is available
openai_client = None
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Configuration
NUM_SAMPLES = 25  # Number of new samples to generate per model
START_IDX = 50   # Start at sample_051.txt (0-based index)

# Only use the new prompts for this run
SAMPLE_PROMPTS = [
    "Explain the concept of neural networks in simple terms.",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe how attention mechanisms work in transformer models.",
    "What is the role of activation functions in neural networks?",
    "Explain the concept of backpropagation.",
    "What are the advantages and disadvantages of deep learning?",
    "How does transfer learning work in machine learning?",
    "Explain the concept of regularization in neural networks.",
    "What is the role of batch normalization in deep learning?",
    "Describe how convolutional neural networks process images.",
    "How does gradient descent work, and why is it important in machine learning?",
    "What are the key differences between a CPU and a GPU?",
    "Compare relational and non-relational databases with examples.",
    "Explain how hashing functions work in cryptography.",
    "What is the difference between compile-time and run-time errors?",
    "How do blockchain consensus mechanisms like Proof of Work and Proof of Stake differ?",
    "What is a race condition in concurrent programming, and how can it be prevented?",
    "Describe how a computer boots up from powering on to loading the operating system.",
    "Explain vector embeddings and their role in natural language processing.",
    "What are common techniques for preventing overfitting in machine learning?",
    "What is pipelining in CPU architecture?",
    "How does virtual memory work?",
    "What is the difference between threads and processes?",
    "Explain how cache memory improves performance.",
    "How do interrupts work in an operating system?",
    "How do decision trees make predictions?",
    "What is the difference between precision and recall in classification tasks?",
    "Explain how reinforcement learning differs from supervised learning.",
    "How does the K-means clustering algorithm work?",
    "What is the vanishing gradient problem in deep learning?",
    "Compare L1 and L2 regularization and when to use each.",
    "How does early stopping help prevent overfitting?",
    "What is the difference between generative and discriminative models?",
    "How does a random forest improve upon a single decision tree?",
    "Explain the role of the softmax function in neural networks.",
    "How does DNS resolve a domain name?",
    "What is the difference between TCP and UDP?",
    "How does HTTPS secure communication?",
    "What is the purpose of a load balancer in web architecture?",
    "Explain how IP addresses and subnetting work.",
    "How does the quicksort algorithm work?",
    "What is a hash table and how is it used?",
    "Explain the difference between depth-first search and breadth-first search.",
    "What is dynamic programming and when should it be used?",
    "Describe the concept of Big-O notation.",
    "How does a relational database enforce data integrity?",
    "What are the main differences between REST and GraphQL?",
    "What is the SOLID principle in object-oriented design?",
    "Explain the concept of version control and how Git supports it.",
    "What is a container, and how does Docker work?",
    "What is a memory leak and how can it be avoided?",
    "Explain the difference between symmetric and asymmetric encryption.",
    "How does a compiler translate source code into machine code?",
    "What is the difference between a stack and a queue?",
    "Explain the concept of recursion and when it should be used.",
    "How does garbage collection work in programming languages like Java or Python?",
    "What is a microservice architecture and what are its benefits?",
    "Explain the role of APIs in software development.",
    "What are the key differences between HTTP and WebSocket protocols?",
    "How does OAuth 2.0 work for authentication?",
    "What is a deadlock in operating systems and how can it be prevented?",
    "How do you design a scalable system for millions of users?",
    "Explain how MapReduce works and give an example use case.",
    "What is eventual consistency in distributed systems?",
    "How does a CDN (Content Delivery Network) improve web performance?",
    "What is the role of the ARP protocol in networking?",
    "Explain how RAID works and what levels like RAID 0, 1, and 5 mean.",
    "How does a binary search algorithm work and what is its time complexity?",
    "What is an AVL tree and how does it maintain balance?",
    "Explain the CAP theorem in distributed databases.",
    "What are the benefits and drawbacks of NoSQL databases?",
    "How does an operating system schedule tasks on a multi-core CPU?",
    "Explain what containers are and how they differ from virtual machines.",
    "How does a TLS handshake establish a secure connection?",
    "What are software design patterns and why are they useful?"
][START_IDX:START_IDX+NUM_SAMPLES]

def setup_directories():
    """Ensure all required directories exist."""
    base_dir = Path('data/raw')
    for model in ['gpt4', 'claude', 'llama']:
        (base_dir / model).mkdir(parents=True, exist_ok=True)

def generate_gpt4_response(prompt):
    """Generate response using GPT-4 Turbo."""
    if not OPENAI_API_KEY or not openai_client:
        print("Skipping GPT-4 generation: No API key provided")
        return "No OpenAI API key provided. This is a placeholder response."
        
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-0125-preview",  # Changed to GPT-4 Turbo
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant explaining technical concepts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating GPT-4 response: {str(e)}")
        return f"GPT-4 API error: {str(e)[:200]}..."

def generate_claude_response(prompt):
    """Generate response using Claude."""
    if not ANTHROPIC_API_KEY:
        print("Skipping Claude generation: No API key provided")
        return "No Anthropic API key provided. This is a placeholder response."
        
    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=500,
            system="You are a helpful AI assistant explaining technical concepts.",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error generating Claude response: {str(e)}")
        return f"Claude API error: {str(e)[:200]}..."

def generate_llama_response(prompt):
    """Generate response using LLaMA via llmapi.com."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}"
        }
        
        payload = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant explaining technical concepts."},
                {"role": "user", "content": prompt}
            ]
        }
        
        # Use the correct endpoint URL we discovered
        url = LLAMA_API_ENDPOINT.rstrip('/') + "/chat/completions"
        
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        else:
            error_msg = f"LLaMA API error: Status {response.status_code}, Response: {response.text}"
            print(error_msg)
            return error_msg
    except Exception as e:
        print(f"Error generating LLaMA response: {str(e)}")
        return f"LLaMA API error: {str(e)[:200]}..."

def save_response(model, sample_num, response):
    """Save the response to a file."""
    if response:
        filename = f"data/raw/{model}/sample_{sample_num:03d}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"Saved {filename}")
    else:
        print(f"Failed to save response for {model} sample {sample_num}")

def main():
    # Setup directories
    setup_directories()
    
    # Generate samples for each model
    for i, prompt in enumerate(SAMPLE_PROMPTS, START_IDX+1):
        print(f"\nGenerating sample {i} for all models...")
        
        # Generate and save GPT-4 response
        print("Generating GPT-4 response...")
        gpt4_response = generate_gpt4_response(prompt)
        save_response('gpt4', i, gpt4_response)
        time.sleep(1)  # Rate limiting
        
        # Generate and save Claude response
        print("Generating Claude response...")
        claude_response = generate_claude_response(prompt)
        save_response('claude', i, claude_response)
        time.sleep(1)  # Rate limiting
        
        # Generate and save LLaMA response
        print("Generating LLaMA response...")
        llama_response = generate_llama_response(prompt)
        save_response('llama', i, llama_response)
        time.sleep(1)  # Rate limiting

if __name__ == "__main__":
    main() 