from flask import Flask, request, jsonify
import faiss
import numpy as np
import openai

app = Flask(__name__)

# Load the FAISS index and metadata
INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "metadata.json"

# Load FAISS index
dimension = 1536  # Replace with your actual embedding dimension
index = faiss.IndexFlatL2(dimension)
faiss.read_index(INDEX_FILE)

# Load metadata
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

# OpenAI API key (required for embeddings if using OpenAI)
openai.api_key = "your_openai_api_key"

# Function to get embeddings for a query
def get_query_embedding(query):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    return np.array(response["data"][0]["embedding"])

# Endpoint to fetch query results
@app.route('/query', methods=['POST'])
def query_index():
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 5)  # Number of results to return

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Get the embedding for the query
        query_embedding = get_query_embedding(query).astype('float32')

        # Search the FAISS index
        distances, indices = index.search(np.array([query_embedding]), top_k)

        # Fetch metadata for the top results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata):
                results.append({
                    "rank": i + 1,
                    "score": float(distances[0][i]),
                    "data": metadata[idx]
                })

        return jsonify({"query": query, "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
