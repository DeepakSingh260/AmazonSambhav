from flask import Flask, request, jsonify
import numpy as np
import openai
from RAG_Model import call_model
app = Flask(__name__)


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
        response , citations = call_model(query)

        return jsonify({"query": query, "results": response, "citations":citations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
