import streamlit as st
import psycopg2
import os
from datetime import datetime
import pytz # Import pytz for timezone awareness

# --- Database Connection ---
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "news_sentiment_db"), # Default if not set, but better to set via env var
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost") # Default to localhost if not specified
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# --- Fetch Articles from Database ---
def get_articles_from_db(conn, limit=100):
    articles = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id, headline, url, content, publication_date, 
                    sentiment_label, sentiment_score, is_user_preferred,
                    processed_at
                FROM articles
                ORDER BY publication_date DESC
                LIMIT %s;
            """, (limit,))
            # Fetch all rows from the query
            rows = cur.fetchall()
            # Get column names from cursor description
            cols = [desc[0] for desc in cur.description]
            
            # Convert rows to list of dictionaries for easier handling
            for row in rows:
                articles.append(dict(zip(cols, row)))
    except Exception as e:
        st.error(f"Error fetching articles from database: {e}")
    return articles

# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use wide layout for better display of data
st.title("📰 Real-time News Dashboard")

# Display Last Processed Timestamp (from your existing logic, if needed)
# For now, let's just display what's in the DB
st.write("---")

# Get DB connection
conn = get_db_connection()

if conn:
    st.subheader("Latest Articles")
    articles = get_articles_from_db(conn, limit=50) # Fetch up to 50 latest articles

    if articles:
        # Create a dataframe for better display in Streamlit
        import pandas as pd
        df = pd.DataFrame(articles)

        # Basic data cleanup for display
        df['publication_date'] = df['publication_date'].dt.strftime('%Y-%m-%d %H:%M')
        df['processed_at'] = df['processed_at'].dt.strftime('%Y-%m-%d %H:%M')

        # Order columns for display
        display_cols = [
            'headline', 'sentiment_label', 'sentiment_score', 'publication_date',
            'is_user_preferred', 'url' 
        ]
        
        # Display the DataFrame
        st.dataframe(df[display_cols], use_container_width=True, height=600)

        st.write("---")
        st.info(f"Displaying {len(articles)} latest articles from your database.")

    else:
        st.warning("No articles found in the database. Run your news agent script to populate it!")

    # Close the database connection
    conn.close()