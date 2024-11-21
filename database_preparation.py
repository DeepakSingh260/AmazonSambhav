import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import faiss
import numpy as np

# Load data from the JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

# Parse and preprocess
def preprocess_data(data):
    documents = []
    for item in data:
        text = item.get("data", "")
        meta = {
            "url": item.get("link", ""),
            "type_of_data": item.get("type_of_data", "")
        }
        documents.append({"text": text, "metadata": meta})
    return documents

data = load_data("scraped_data.json")
documents = preprocess_data(data)# Split documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunked_docs = []

for doc in documents:
    for chunk in text_splitter.split_text(doc["text"]):
        chunked_docs.append({"text": chunk, "metadata": doc["metadata"]})


# Initialize embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create embeddings for all chunks
embeddings = []
for doc in chunked_docs:
    embedding = embedding_model.embed_query(doc["text"])
    embeddings.append({"embedding": embedding, "metadata": doc["metadata"]})



dimension = len(embeddings[0]["embedding"])
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
vectors = np.array([e["embedding"] for e in embeddings])
index.add(vectors)

# Save metadata separately
metadata = [e["metadata"] for e in embeddings]

# Save index and metadata to disk
faiss.write_index(index, "vector_index.faiss")
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)