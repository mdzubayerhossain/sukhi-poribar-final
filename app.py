from flask import Flask, render_template, request, jsonify
import time
import numpy as np
import faiss
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_community.document_loaders import TextLoader
import threading
from itertools import cycle
from cachetools import TTLCache
from queue import Queue

# Load environment variables
load_dotenv()

# API tokens and limits setup
apis = [
    {"api_key": "jm933nnK6gdjUXGr4gIVmjPwRVMFELcp", "minute_limit": 500000, "monthly_limit": 100000000, "used_in_last_minute": 0, "used_in_month": 0},
    {"api_key": "t0zJXzm2HklV9YgNBkjHyCC9z775RYAM", "minute_limit": 500000, "monthly_limit": 100000000, "used_in_last_minute": 0, "used_in_month": 0},
    {"api_key": "wtiP9b9HFvZdbyhTByQLjLeAXctbzp3F", "minute_limit": 500000, "monthly_limit": 100000000, "used_in_last_minute": 0, "used_in_month": 0},
    # Add more API keys as needed
]

api_cycle = cycle(apis)

# Function to get the next available API
def get_next_api():
    current_time = time.time()
    for api in api_cycle:
        if (
            api["used_in_last_minute"] < api["minute_limit"]
            and api["used_in_month"] < api["monthly_limit"]
        ):
            api["used_in_last_minute"] += 1
            api["used_in_month"] += 1
            return Mistral(api_key=api["api_key"])
    raise Exception("All APIs have exceeded their limits.")

# Reset usage statistics every minute
def reset_minute_usage():
    while True:
        time.sleep(60)  # Reset every minute
        for api in apis:
            api["used_in_last_minute"] = 0

# Reset usage statistics every month
def reset_monthly_usage():
    while True:
        time.sleep(30 * 24 * 60 * 60)  # Reset every month
        for api in apis:
            api["used_in_month"] = 0

# Start the background tasks for resetting usage
threading.Thread(target=reset_minute_usage, daemon=True).start()
threading.Thread(target=reset_monthly_usage, daemon=True).start()

# Load data
loader = TextLoader(r"D:\Coding\User Handle with Mistral\book.txt", encoding="utf-8")
docs = loader.load()
text = docs[0].page_content

# Chunk text data
chunk_size = 6500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to get text embedding with error handling
def get_text_embedding(input_text):
    try:
        client = get_next_api()  # Use the next available API
        embeddings_batch_response = client.embeddings.create(
            model="mistral-embed",
            inputs=[input_text]  # Ensure the input is a list of strings
        )
        return embeddings_batch_response.data[0].embedding
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        raise

# Add a delay between API calls to avoid rate limiting
delay_seconds = 2
text_embeddings = []
for chunk in chunks:
    embedding = get_text_embedding(chunk)
    text_embeddings.append(embedding)
    time.sleep(delay_seconds)

# Convert embeddings to a NumPy array and index with Faiss
text_embeddings = np.array(text_embeddings)
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

# Cache for storing responses
cache = TTLCache(maxsize=100, ttl=300)

# Request queue for managing simultaneous requests
request_queue = Queue()

# Function to generate a response with Mistral
def run_mistral(prompt, model="open-mistral-nemo"):
    client = get_next_api()  # Use the next available API
    messages = [{"role": "user", "content": prompt}]
    time.sleep(delay_seconds)
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content

# Function to process requests sequentially
def process_requests():
    while True:
        question, callback = request_queue.get()
        try:
            # Check if the response is in the cache
            if question in cache:
                answer = cache[question]
            else:
                # Get embedding for the question
                question_embedding = np.array([get_text_embedding(question)])

                # Find the closest matching chunk
                D, I = index.search(question_embedding, k=2)
                retrieved_chunk = [chunks[i] for i in I[0]]

                # Construct the prompt for Mistral
                prompt = f"""
                Context information is below.
                ---------------------
                {retrieved_chunk}
                ---------------------
                আমাকে সব সময় বাংলা উত্তর দিবে আমার দেওয়া তথ্যের উপর ভিত্তি করে যদি আমার দেওয়া তথ্যের ভিতর না থাকে তোমার মত করে উত্তর দিত্ত।
                Query: {question}
                Answer:
                """

                # Get response from Mistral
                answer = run_mistral(prompt)
                cache[question] = answer  # Store the response in the cache

            # Return the response through the callback
            callback(answer)
        except Exception as e:
            callback(str(e))
        finally:
            request_queue.task_done()

# Start the background task for processing requests
threading.Thread(target=process_requests, daemon=True).start()

# Flask setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    question = request.args.get('msg')
    response_holder = []

    # Define a callback to store the response
    def callback(response):
        response_holder.append(response)

    # Add the request to the queue
    request_queue.put((question, callback))

    # Wait for the response to be processed
    request_queue.join()

    # Return the response
    return jsonify(response_holder[0])

if __name__ == "__main__":
    app.run(debug=True)