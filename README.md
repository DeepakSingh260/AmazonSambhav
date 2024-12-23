﻿# AmazonSambhav
# **Export Compliance and Incentive Information Aggregator**

This project aggregates information from various government and third-party sources to provide a comprehensive overview of:  
1. **Compliance requirements for exporting to the US and Europe**  
2. **Relevant incentives and grants available for exporters in the US and Europe**

The system includes a web scraper, vectorized storage for efficient similarity search, and a Flask API for querying the database.

---

## **How It Works**

### **1. Web Scraping**
The script uses a queue-based approach to crawl and collect data from predefined URLs, along with their metadata. Key features include:
- **Metadata Tracking**: Each URL is associated with metadata (e.g., data type and depth).
- **Depth Control**: Scraping is performed up to a maximum depth of 4, ensuring broad but limited crawling.
- **Visited Links Tracking**: All visited URLs are stored in a JSON file to avoid duplication.

### **2. Data Processing**
- Extracted data is saved in JSON format with metadata.
- Each piece of data is classified based on its source (e.g., HS Code, Compliance (EU), Incentive (US)).

### **3. Vectorization**
The content is vectorized using OpenAI's `text-embedding-ada-002` model:
- Each document's embedding is stored in a **FAISS index**.
- Corresponding metadata (e.g., URL, type of data, content snippet) is saved in a separate JSON file.

### **4. Flask API**
A Flask API provides an interface to query the vector database:
- **Input**: User query in natural language.
- **Output**: Top-matching documents with scores and metadata.

---

## **Technologies Used**
- **Python Libraries**:
  - `requests`, `BeautifulSoup`: For web scraping.
  - `faiss`: For vector storage and similarity search.
  - `openai`: For generating text embeddings.
  - `flask`: For building the API.
- **FAISS**: Vector database for efficient similarity search.
- **OpenAI API**: To generate text embeddings.

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.8 or later
- OpenAI API Key
- Installed Python libraries:
  ```bash
  pip install requests beautifulsoup4 faiss-cpu flask openai
  ```

### **2. Run the Web Scraper**
1. Update the `base_urls` in the script to include the sources to scrape.
2. Run the scraper to collect data:
   ```bash
   python database_creation.py
   ```
3. This will:
   - Save scraped data to `scraped_data.json`.
   - Save visited links to `visited_links.json`.

### **3. Build the Vector Index**
1. Process the scraped data to generate embeddings:
   ```bash
   python data_embedding.py
   ```
2. This will:
   - Create a FAISS index (`vector_index.faiss`).
   - Save metadata to `metadata.json`.
  
     
### **3.5 Run RAG Model **

 ```bash
   python RAG_Model.py
   ```


### **4. Run the Flask API**
1. Start the API server:
   ```bash
   python url.py
   ```
2. The server will be accessible at `http://127.0.0.1:5000`.

---


## **Adding New Data Sources**
not yet implemented
1. Add new URLs to the `base_urls` list in `database_creation.py`, with appropriate metadata.
2. Re-run the scraper and rebuild the vector index.

---

## **Use Cases**
- **Exporters**: Quickly access compliance requirements for exporting to specific regions.
- **Policy Analysts**: Explore incentive and grant information for exporters in the US and Europe.
- **Trade Consultants**: Provide tailored advice to clients using relevant data.

---

## **Future Enhancements**
- Implement additional natural language processing for better query interpretation.
- Integrate multiple embedding models for improved semantic search.
- Add support for real-time updates to the vector database.

---

Feel free to contribute by submitting pull requests or opening issues! 🚀
