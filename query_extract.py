import spacy
from sentence_transformers import SentenceTransformer
# Load NLP model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")
def parse_user_query(query):
    """
    Extracts topics, keywords, and regions from the user's query using NLP.
    """
    doc = nlp(query)

    # Define categories
    topics = {"duty", "tariff", "compliance", "policy", "regulation"}
    regions = {"india", "eu", "europe", "usa", "canada", "china", "uk"}

    # Extract entities and keywords
    entities = [ent.text.lower() for ent in doc.ents]
    keywords = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    
    # Match topics and regions
    matched_topics = set(keywords) & topics
    matched_regions = set(keywords) & regions
    
    parsed_query =  {
        "topics": list(matched_topics),
        "keywords": keywords,
        "regions": list(matched_regions),
        "entities": entities
    }
    query_text = " ".join(parsed_query["topics"] + parsed_query["keywords"] + parsed_query["regions"]+ parsed_query["entities"])
    return query_text
# Example user query
# user_query = "What is the duty for cuttlefish imports in the EU?"
# parsed_query = parse_user_query(user_query)
# query_vector = model.encode(query_text)
# print(query_vector)
