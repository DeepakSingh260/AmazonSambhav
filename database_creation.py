import requests
from bs4 import BeautifulSoup
import json
import time
from collections import deque

# Initialize queue
link_queue = deque()

# Define maximum depth
MAX_DEPTH = 5

# Base URLs with initial metadata
base_urls = [
    {"url": "https://www.cybex.in/hs-codes/", "data_type": "HS Code", "depth": 0},
    {"url": "https://www.dgft.gov.in/cp", "data_type": "Compliance (India)", "depth": 0},
    {"url": "https://trade.ec.europa.eu/access-to-markets/en/home", "data_type": "Compliance (EU)", "depth": 0},
    {"url": "https://www.trade.gov", "data_type": "Compliance (US)", "depth": 0},
    {"url": "https://www.usacompetes.com", "data_type": "Incentive (US)", "depth": 0},
    {"url": "https://www.edc.ca", "data_type": "Incentive (Canada)", "depth": 0},
]

# Add base URLs to the queue
for base_url in base_urls:
    link_queue.append(base_url)

# Function to extract and classify data
def extract_and_classify_data(url, data_type):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract page content
        page_content = soup.get_text(strip=True)
        
        # Return structured data
        return {
            "link": url,
            "data": page_content,  # Store first 500 characters
            "type_of_data": data_type
        }
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# Set of visited URLs to avoid duplicates
visited_urls = set()

# Load existing visited links if available
try:
    with open('visited_links.json', 'r') as f:
        visited_urls = set(json.load(f))
except FileNotFoundError:
    visited_urls = set()

# Process the queue
while link_queue:
    # Pop the first item from the queue
    current = link_queue.popleft()
    current_url = current["url"]
    current_depth = current["depth"]
    current_data_type = current["data_type"]

    # Skip if URL has already been visited
    if current_url in visited_urls:
        continue
    visited_urls.add(current_url)

    print(f"Processing {current_url} at depth {current_depth}")

    # Stop processing if depth exceeds the maximum
    if current_depth > MAX_DEPTH:
        continue

    # Extract and classify data
    extracted_data = extract_and_classify_data(current_url, current_data_type)
    if extracted_data:
        # Save to JSON file
        with open('scraped_data.json', 'a') as f:
            json.dump(extracted_data, f, indent=4)
            f.write('\n')

    # Extract links and add them to the queue
    try:
        response = requests.get(current_url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]

        for link in links:
            # Convert relative URLs to absolute
            full_url = link if link.startswith("http") else f"{current_url.rstrip('/')}/{link.lstrip('/')}"
            
            # Avoid adding already visited URLs
            if full_url not in visited_urls:
                link_queue.append({"url": full_url, "data_type": current_data_type, "depth": current_depth + 1})
    except Exception as e:
        print(f"Error extracting links from {current_url}: {e}")

    # Save visited links to JSON file
    with open('visited_links.json', 'w') as f:
        json.dump(list(visited_urls), f, indent=4)

    # Politeness delay
    time.sleep(1)

print("Scraping complete. Data saved to 'scraped_data.json' and 'visited_links.json'.")
