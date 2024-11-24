import faiss
import json
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from query_extract import parse_user_query
client = OpenAI(api_key = "api_key_here")

# Load the FAISS index
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

# Load the metadata
def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)

# Embed the query
def embed_query(query, embedding_model):
    return np.array(embedding_model.embed_query(query)).astype('float32')

# Search the FAISS index
def search_faiss_index(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]

# Query LLM to analyze data and generate an answer
def query_llm(question, context):
    llm_input = (
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        "Answer as best as you can based on context"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes data and answers queries."},
            {"role": "user", "content": llm_input}
        ]
    )
    return response

#query formatted json for context
def fetch_context(indices, context_data):
    contexts = []
    for idx in indices:
        # Ensure the index is within range
        if idx < len(context_data):
            contexts.append(context_data[idx].get("data", "No context available"))
        else:
            contexts.append("No context available")
    return contexts

# Perform the query and return results
def query_model(query, index, metadata, embedding_model, k=5, context_data = ""):
    query_embedding = embed_query(query, embedding_model).reshape(1, -1)
    distances, indices = search_faiss_index(index, query_embedding, k)
    contexts = fetch_context(indices, context_data)
    results = []
    for i, idx in enumerate(indices):
        if idx < len(metadata):  # Ensure index is within metadata bounds
            meta = metadata[idx]
            context = meta.get("data", "")
            citation = meta.get("url", "")
            print("context",contexts[i])
            llm_answer = query_llm(query, contexts[i])
            results.append({
                "metadata": meta,
                "distance": distances[i],
                "answer": llm_answer.choices[0].message.content,
                "citation": citation
            })
    return results
def call_model(query):
    # Paths to the index and metadata files
    index_path = "vector_index.faiss"
    metadata_path = "metadata.json"

    # Load the index and metadata
    faiss_index = load_faiss_index(index_path)
    metadata = load_metadata(metadata_path)

    # Initialize the embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key = "api_key_here")

    # Input query
    query_text = parse_user_query(query)
    context_data_path = "formatted_scraped_data.json"
    with open(context_data_path, 'r') as f:
        context_data = [json.loads(line) for line in f]
    # Perform the query
    k = 1  # Number of nearest neighbors to retrieve
    results = query_model(query_text, faiss_index, metadata, embedding_model, k, context_data)

    # Print the results
    print(f"Query: {query_text}")
    print("Results:")
    for result in results:
        print(f"Answer: {result['answer']}")
        print(f"Citation: {result['citation']}")
        print(f"Distance: {result['distance']:.4f}\n")
    return results[0]['answer'] , results[0]['citation']
