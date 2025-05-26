import requests
import time
import streamlit as st
import pytz
import json
import psycopg2
from psycopg2 import sql
from sentence_transformers import SentenceTransformer
from ollama import Client as OllamaClient
import os
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
# import xml.etree.ElementTree as ET # Removed if no sitemap parsing

# --- Database Connection ---
def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.
    Reads credentials from environment variables, with macOS Homebrew defaults if not set.
    """
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME', 'news_sentiment_db'),
        user=os.getenv('DB_USER', "bbhavna"), # Default to macOS username
        # For Homebrew on macOS, a password often isn't required for local connections.
        # Uncomment and provide if you've set one for your Postgres user.
        # password=os.getenv('DB_PASSWORD', ''),
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432')
    )
    return conn

def get_last_processed_timestamp(conn):
    """Retrieves the timestamp of the last processed article from the DB."""
    with conn.cursor() as cur:
        # CRUCIAL CHANGE: Get the MAX(publication_date) from the articles table
        # This makes sure we only fetch articles published AFTER the latest one already saved.
        cur.execute("SELECT MAX(publication_date) FROM articles;")
        last_timestamp = cur.fetchone()[0]

    if last_timestamp:
        # Ensure it's UTC-aware for comparison, as Bright Data's dates are UTC.
        return last_timestamp.replace(tzinfo=pytz.utc)
    else:
        # If no articles have been processed yet (table is empty),
        # return the absolute minimum datetime, also UTC-aware.
        # This will ensure all incoming articles are treated as "new" initially.
        return datetime.min.replace(tzinfo=pytz.utc)

def fetch_urls_from_rss(rss_feed_url, limit=5):
    """
    Fetches the latest article URLs from a given RSS feed.
    """
    article_urls = []
    try:
        response = requests.get(rss_feed_url, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        root = ET.fromstring(response.content)

        # RSS feeds typically have 'item' tags for each article
        for item in root.findall('.//item'):
            link = item.find('link')
            title = item.find('title') # We can also get the title for keyword if needed

            if link is not None and link.text:
                url = link.text.strip()
                # Basic validation: ensure it looks like an article URL
                if url.startswith('http') and '/articles/' in url:
                    # For simplicity, we'll assign a placeholder keyword
                    # In a real scenario, you might derive keywords from the title/description
                    article_urls.append({"url": url, "keyword": title.text.strip() if title is not None else ""})
                if len(article_urls) >= limit:
                    break # Stop after collecting the desired number of URLs

    except requests.exceptions.RequestException as e:
        print(f"Error fetching RSS feed {rss_feed_url}: {e}")
    except ET.ParseError as e:
        print(f"Error parsing RSS feed XML {rss_feed_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing RSS feed {rss_feed_url}: {e}")

    return article_urls
# --- AI Model Initialization ---
embedding_model = SentenceTransformer('all-mpnet-base-v2')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
ollama_client = OllamaClient(host=OLLAMA_HOST)

# --- Bright Data Configuration (CRITICAL: REPLACE THESE VALUES) ---
BRIGHTDATA_API_TOKEN = os.getenv('BRIGHTDATA_API_TOKEN', 'YOUR_ACTUAL_BRIGHTDATA_API_TOKEN_HERE')
BRIGHTDATA_DATASET_ID = os.getenv('BRIGHTDATA_DATASET_ID', 'YOUR_BRIGHTDATA_DATASET_ID_HERE')
BRIGHTDATA_TRIGGER_URL = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={BRIGHTDATA_DATASET_ID}&include_errors=true"

# This is the correct URL structure for fetching a specific snapshot's status and results
BRIGHTDATA_SNAPSHOT_FETCH_URL = "https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"


def fetch_new_articles_from_brightdata(last_processed_timestamp):
    """
    Triggers Bright Data collection, waits for results, and returns newly fetched articles.
    """
    if not BRIGHTDATA_API_TOKEN or BRIGHTDATA_API_TOKEN == 'YOUR_ACTUAL_BRIGHTDATA_API_TOKEN_HERE':
        print("ERROR: Bright Data API Token is not set or is still the placeholder. Cannot fetch data.")
        return []
    if not BRIGHTDATA_DATASET_ID or BRIGHTDATA_DATASET_ID == 'YOUR_BRIGHTDATA_DATASET_ID_HERE':
        print("ERROR: Bright Data Dataset ID is not set or is still the placeholder. Cannot fetch data.")
        return []

    # --- Providing Hardcoded "Seed" URLs for Trigger ---
    BBC_NEWS_RSS_URL = "http://feeds.bbci.co.uk/news/rss.xml" # You can change this RSS feed URL
    
    print(f"Fetching latest URLs from RSS feed: {BBC_NEWS_RSS_URL}")
    # Fetch, for example, the latest 10 articles from the RSS feed
    payload_data = fetch_urls_from_rss(BBC_NEWS_RSS_URL, limit=10) # <--- THIS IS THE NEW LINE

    # --- These lines remain as they correctly prepare data for the Bright Data trigger ---
    if not payload_data:
        print("No URLs specified to trigger Bright Data collection. Exiting fetch.")
        return []

    headers = {
        "Authorization": f"Bearer {BRIGHTDATA_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = json.dumps(payload_data)

    print(f"Triggering Bright Data collection for {len(payload_data)} unique URLs...")
    try:
        trigger_response = requests.post(BRIGHTDATA_TRIGGER_URL, headers=headers, data=payload)
        trigger_response.raise_for_status()
        raw_trigger_response_json = trigger_response.json()
        print(f"DEBUG: Raw Trigger Response JSON: {raw_trigger_response_json}")
        snapshot_id = raw_trigger_response_json.get('snapshot_id')
        print(f"Collection triggered. Snapshot ID: {snapshot_id}")
    except requests.exceptions.RequestException as e:
        print(f"Error triggering Bright Data: {e}")
        return []

    if not snapshot_id:
        print("Failed to get snapshot ID from Bright Data trigger response.")
        return []

    # --- Poll for Data Availability ---
    # We poll the snapshot URL until it returns a non-empty list of articles.
    # This implies the collection is 'Ready' and the data is available.
    articles_collected = []
    retries = 0
    max_retries = 60 # Poll for up to 10 minutes (60 * 10 seconds)
    polling_interval = 10 # seconds

    print("Polling Bright Data for collection results...")
    polling_snapshot_url = BRIGHTDATA_SNAPSHOT_FETCH_URL.format(snapshot_id=snapshot_id)

    # Loop until articles are collected (list is non-empty) or max_retries reached
    while not articles_collected and retries < max_retries:
        try:
            print(f"DEBUG: Polling Snapshot URL: {polling_snapshot_url}")
            progress_response = requests.get(polling_snapshot_url, headers={"Authorization": f"Bearer {BRIGHTDATA_API_TOKEN}"})
            progress_response.raise_for_status() # Raise HTTPError for bad responses (e.g., 404, 500)

            response_content = progress_response.json()
            print(f"DEBUG: Raw Progress Response JSON: {response_content}")

            # If the response is a list and contains items, then data is ready
            if isinstance(response_content, list) and len(response_content) > 0:
                articles_collected = response_content # Data is available!
                print(f"Collection 'Ready'. Fetched {len(articles_collected)} raw articles for Snapshot ID: {snapshot_id}")
                break # Exit the loop as data is found.
            elif isinstance(response_content, list) and len(response_content) == 0:
                # This indicates the snapshot is still processing or yielded no results yet
                print(f"Snapshot ID {snapshot_id} is processing (no data yet). Retrying... (Retry {retries+1}/{max_retries})")
            else:
                # Handle unexpected formats (e.g., if it returns a dict with 'status' when not ready)
                # This path should ideally not be taken if the API consistently returns a list.
                # If it's a dict, try to find a 'status' or 'result' key
                status_from_response = response_content.get('status') if isinstance(response_content, dict) else None
                if status_from_response:
                    print(f"Snapshot ID {snapshot_id} status: {status_from_response}. Retrying... (Retry {retries+1}/{max_retries})")
                    if status_from_response.lower() == "ready" and isinstance(response_content, dict):
                        # If it's ready AND a dict, try to get the 'result' key.
                        articles_collected = response_content.get('result', [])
                        if articles_collected: # If results found, break
                            print(f"Collection 'Ready' from status dict. Fetched {len(articles_collected)} raw articles for Snapshot ID: {snapshot_id}")
                            break
                else:
                    print(f"Unexpected empty or non-list response format for Snapshot ID {snapshot_id}. Retrying... (Retry {retries+1}/{max_retries})")

            time.sleep(polling_interval)
            retries += 1

        except requests.exceptions.RequestException as e:
            # Catches network errors, 4xx/5xx HTTP errors (like 404 if snapshot_id is bad)
            print(f"Error polling Bright Data snapshot {snapshot_id}: {e}. Retrying... (Retry {retries+1}/{max_retries})")
            time.sleep(polling_interval)
            retries += 1
        except json.JSONDecodeError as e:
            # Catches cases where response is not valid JSON
            print(f"Error decoding JSON response from Bright Data for snapshot {snapshot_id}: {e}. Retrying...")
            time.sleep(polling_interval)
            retries += 1

    # Check if data was actually collected after the loop finishes
    if not articles_collected:
        print(f"Bright Data collection timed out or failed to retrieve data for snapshot ID {snapshot_id}.")
        return []

    # --- Data Filtering (using articles_collected directly) ---
    # raw_articles_data is now 'articles_collected' which already contains the list of articles
    fetched_articles = []
    for article in articles_collected: # Iterate directly over the collected list
        try:
            pub_date_str = article.get('publication_date')
            if pub_date_str:
                pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                if pub_date > last_processed_timestamp:
                    fetched_articles.append(article)
            else:
                print(f"Warning: Article {article.get('id')} has no publication_date. Appending without timestamp check.")
                fetched_articles.append(article)
        except ValueError as e:
            print(f"Error parsing date for article {article.get('id')}: {e}. Appending article.")
            fetched_articles.append(article)

    print(f"Fetched {len(fetched_articles)} articles from Bright Data for snapshot {snapshot_id}.")
    return fetched_articles

# --- Article Processing and Storage (remains the same) ---
def get_user_preferences(conn, user_id='default_user'):
    """Retrieves user preferences from the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT min_positive_score, max_negative_score, exclude_topics, include_keywords FROM user_preferences WHERE user_id = %s;", (user_id,))
        prefs = cur.fetchone()
        if prefs:
            return {
                'min_positive_score': prefs[0],
                'max_negative_score': prefs[1],
                'exclude_topics': prefs[2] if prefs[2] else [],
                'include_keywords': prefs[3] if prefs[3] else []
            }
        print(f"User preferences not found for user_id: {user_id}. Using default logical values.")
        return {
            'min_positive_score': 0.5,
            'max_negative_score': -0.5,
            'exclude_topics': [],
            'include_keywords': []
        }

def process_and_store_article(conn, article_data, user_preferences):
    """Processes a single article and stores it in the database."""
    article_id = article_data.get('id')
    url = article_data.get('url')
    headline = article_data.get('headline')
    content = article_data.get('content')
    publication_date_str = article_data.get('publication_date')

    if not all([article_id, url, headline, content, publication_date_str]):
        print(f"Skipping article {article_id}: Missing essential data.")
        return

    try:
        publication_date = datetime.fromisoformat(publication_date_str.replace('Z', '+00:00'))
    except ValueError as e:
        print(f"Skipping article {article_id}: Invalid publication_date format - {e}")
        return

    try:
        embedding = embedding_model.encode(content).tolist()
    except Exception as e:
        print(f"Error generating embedding for {article_id}: {e}")
        embedding = None

    sentiment_label = "NEUTRAL"
    sentiment_score = 0.0
    try:
        llm_prompt = (
            f"Analyze the sentiment of the following news article content. "
            f"Respond with only two comma-separated values: the sentiment word (POSITIVE, NEGATIVE, or NEUTRAL) "
            f"and a numerical score from -1.0 (very negative) to 1.0 (very positive). "
            f"Example output: 'POSITIVE, 0.8'\n\nContent: {content}"
        )
        response = ollama_client.generate(model='llama3.2', prompt=llm_prompt)
        raw_output = response['response'].strip()

        parts = raw_output.split(',')
        if len(parts) == 2:
            label = parts[0].strip().upper()
            score_str = parts[1].strip()
            if label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                sentiment_label = label
                sentiment_score = float(score_str)
            else:
                print(f"Ollama output unexpected label format for article {article_id}: '{raw_output}'")
        else:
            print(f"Ollama output unexpected format for article {article_id}: '{raw_output}'")
    except Exception as e:
        print(f"Error during Ollama sentiment analysis for {article_id}: {e}")

    is_user_preferred = True

    article_topics = article_data.get('topics', [])
    if article_topics:
        for ex_topic in user_preferences['exclude_topics']:
            if ex_topic.lower() in [t.lower() for t in article_topics]:
                is_user_preferred = False
                break

    if user_preferences['include_keywords']:
        found_keyword = False
        for inc_keyword in user_preferences['include_keywords']:
            if inc_keyword.lower() in content.lower() or inc_keyword.lower() in headline.lower():
                found_keyword = True
                break
        if not found_keyword:
            is_user_preferred = False

    try:
        with conn.cursor() as cur:
            insert_query = sql.SQL("""
                INSERT INTO articles (id, url, headline, content, publication_date, sentiment_label, sentiment_score, embedding, is_user_preferred)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    url = EXCLUDED.url,
                    headline = EXCLUDED.headline,
                    content = EXCLUDED.content,
                    publication_date = EXCLUDED.publication_date,
                    sentiment_label = EXCLUDED.sentiment_label,
                    sentiment_score = EXCLUDED.sentiment_score,
                    embedding = EXCLUDED.embedding,
                    is_user_preferred = EXCLUDED.is_user_preferred,
                    processed_at = NOW();
            """)
            cur.execute(insert_query, (
                article_id, url, headline, content, publication_date,
                sentiment_label, sentiment_score, embedding, is_user_preferred
            ))
            conn.commit()
            print(f"Article {article_id} processed and stored/updated.")
    except Exception as e:
        conn.rollback()
        print(f"Error storing article {article_id}: {e}")

# --- Main Agent Execution (remains the same) ---
def run_news_agent():
    print("Starting news agent run...")
    conn = None
    try:
        conn = get_db_connection()
        user_prefs = get_user_preferences(conn)

        last_timestamp = get_last_processed_timestamp(conn)
        print(f"Last processed article timestamp: {last_timestamp}")

        new_articles = fetch_new_articles_from_brightdata(last_timestamp)

        if not new_articles:
            print("No new articles to process.")
            return

        for article in new_articles:
            process_and_store_article(conn, article, user_prefs)

        print("News agent run completed successfully.")

    except Exception as e:
        print(f"An error occurred during agent run: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    run_news_agent()