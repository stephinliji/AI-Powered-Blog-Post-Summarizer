import os
import requests
import nltk
from flask import Flask, render_template, request, flash
from newspaper import Article

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Configuration ---
# IMPORTANT: Replace with your Hugging Face API Token
HF_API_TOKEN = "HF_TOKEN"
SUMMARIZATION_MODEL_ID = "facebook/bart-large-cnn"
API_URL_SUMMARIZATION = f"https://api-inference.huggingface.co/models/{SUMMARIZATION_MODEL_ID}"

# --- Helper Function to download NLTK data (if not already present) ---
def download_nltk_data():
    """Checks for required NLTK data packages and downloads them if missing."""
    required_data = ['punkt', 'punkt_tab']
    for resource in required_data:
        try:
            # Check if the resource is available
            nltk.data.find(f'tokenizers/{resource.split("_")[0]}')
            print(f"DEBUG: NLTK '{resource}' resource is already available.")
        except LookupError:
            # Download the resource if it's not found
            print(f"DEBUG: NLTK '{resource}' resource not found. Downloading...")
            nltk.download(resource)
            print(f"DEBUG: NLTK '{resource}' download complete.")

# --- Helper Function to Query Hugging Face Summarization Model ---
def query_summarization_model(text_to_summarize, retries=3, backoff_factor=0.5):
    if not HF_API_TOKEN or "YOUR_NEW_HUGGING_FACE_API_TOKEN" in HF_API_TOKEN:
        return None, "Error: Hugging Face API Token not configured."
        
    # It's good practice to limit the input text length
    max_input_length = 1024 * 5 # Approx 5k characters, adjust as needed
    if len(text_to_summarize) > max_input_length:
        text_to_summarize = text_to_summarize[:max_input_length]
        flash("Input text was truncated to a reasonable length for the model.", "warning")

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": text_to_summarize,
        "parameters": {
            "min_length": 60,
            "max_length": 200, # Max length of the summary
            "do_sample": False
        }
    }
    print(f"DEBUG: Querying Summarization model. Input length: {len(text_to_summarize)}")

    response_obj = None
    for i in range(retries):
        try:
            response_obj = requests.post(API_URL_SUMMARIZATION, headers=headers, json=payload, timeout=60)
            response_obj.raise_for_status()
            result = response_obj.json()
            if isinstance(result, list) and result and "summary_text" in result[0]:
                return result[0]["summary_text"], None
            else:
                return None, "Error: Could not parse model response or no summary found."
        except requests.exceptions.RequestException as e:
            # ... (Error handling and retry logic as in previous apps) ...
            return None, f"Error: Summarization API request failed. {e}"
        except Exception as e:
            return None, f"Error: An unexpected error occurred. {e}"
            
    return None, "Error: Summarization Model query failed after multiple retries."

# --- Helper Function to Extract Text from a URL ---
def get_text_from_url(url):
    print(f"DEBUG: Scraping article from URL: {url}")
    try:
        article = Article(url)
        article.download()
        article.parse()
        if not article.text:
            return None, "Could not extract article text from the URL. The page might be empty or require JavaScript."
        return article.text, None
    except Exception as e:
        print(f"Error scraping URL: {e}")
        return None, f"Failed to process the URL. Please check if it's a valid article URL. Error: {e}"

# --- Main Application Route ---
@app.route('/', methods=['GET', 'POST'])
def summarizer():
    bullet_points = []
    source_info = ""

    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        article_text = ""
        error = None

        if not user_input:
            flash("Please enter a URL or paste some text.", "warning")
        elif user_input.lower().startswith('http'):
            source_info = f"From URL: {user_input}"
            article_text, error = get_text_from_url(user_input)
            if error:
                flash(error, "danger")
        else:
            source_info = "From Pasted Text"
            article_text = user_input
        
        if article_text:
            flash("Content acquired. Generating summary...", "info")
            summary_text, error = query_summarization_model(article_text)
            if error:
                flash(error, "danger")
            elif summary_text:
                # Use NLTK to split the summary into sentences for bullet points
                bullet_points = nltk.sent_tokenize(summary_text)
                flash("Summary generated successfully!", "success")
            else:
                flash("The model did not return a summary.", "warning")
    
    return render_template('summarizer.html', 
                           bullet_points=bullet_points,
                           source_info=source_info)

if __name__ == '__main__':
    # Download NLTK data on first run
    download_nltk_data()
    print("DEBUG: Starting Flask server for Blog Post Summarizer...")
    app.run(debug=True)