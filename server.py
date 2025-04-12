import os
import re
import json
import time
import random
import requests
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from urllib.parse import quote_plus
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp_reddit_agent')

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("RedditReviewAnalysisServer")

# Constants
REDDIT_SEARCH_URL = 'https://www.reddit.com/search.json'
REDDIT_SUBREDDIT_SEARCH_URL = 'https://www.reddit.com/r/{subreddit}/search.json'
REDDIT_POST_URL = 'https://www.reddit.com/comments/{post_id}.json'
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
]

# Helper functions
def get_random_user_agent():
    return random.choice(USER_AGENTS)

def make_reddit_request(url, params=None):
    """Make a request to Reddit API with proper error handling and rate limiting"""
    headers = {
        "User-Agent": get_random_user_agent()
    }
    
    # Add authentication if available
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if client_id and client_secret:
        auth = (client_id, client_secret)
        headers["User-Agent"] = os.getenv("REDDIT_USER_AGENT", "MCP-Reddit-Review-Agent/1.0")
    else:
        auth = None
    
    try:
        response = requests.get(url, params=params, headers=headers, auth=auth, timeout=10)
        
        # Handle rate limiting
        if response.status_code == 429:
            wait_time = int(response.headers.get('Retry-After', 5))
            logger.warning(f"Rate limited by Reddit API. Waiting {wait_time} seconds")
            time.sleep(wait_time)
            return make_reddit_request(url, params)  # Retry
            
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error making Reddit request: {e}")
        return {"error": str(e)}

def extract_post_id(reddit_url):
    """Extract post ID from a Reddit URL"""
    patterns = [
        r'reddit\.com/r/\w+/comments/(\w+)/?',
        r'redd\.it/(\w+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, reddit_url)
        if match:
            return match.group(1)
    return None

# MCP Tool Implementations
@mcp.tool()
def fetch_reviews(query: str, limit: int = 30, timeframe: str = "month", subreddit: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch review posts from Reddit search with comprehensive metadata.
    
    Args:
        query: The search query for product reviews
        limit: Maximum number of results to return (default: 30)
        timeframe: Time range (hour, day, week, month, year, all)
        subreddit: Optional specific subreddit to search
    
    Returns:
        List of review post data including text, metadata, and source info
    """
    search_query = f'"{query}" AND (review OR experience OR recommend OR bought)'
    
    params = {
        "q": search_query,
        "limit": min(limit, 100),  # Reddit caps at 100
        "sort": "relevance",
        "t": timeframe,
        "raw_json": 1
    }
    
    if subreddit:
        url = REDDIT_SUBREDDIT_SEARCH_URL.format(subreddit=subreddit)
    else:
        url = REDDIT_SEARCH_URL
    
    data = make_reddit_request(url, params)
    
    if "error" in data:
        logger.error(f"Error in fetch_reviews: {data['error']}")
        return []
    
    reviews = []
    try:
        for child in data.get("data", {}).get("children", []):
            item = child.get("data", {})
            
            # Skip items without text content
            content = item.get("selftext") or item.get("body") or ""
            if not content.strip():
                continue
                
            # Create comprehensive review object
            review = {
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "text": content,
                "subreddit": item.get("subreddit", ""),
                "score": item.get("score", 0),
                "upvote_ratio": item.get("upvote_ratio", 0.0),
                "num_comments": item.get("num_comments", 0),
                "permalink": f"https://reddit.com{item.get('permalink', '')}",
                "created_utc": item.get("created_utc", 0),
                "is_post": "selftext" in item,
                "author": item.get("author", "[deleted]"),
                "awards": item.get("total_awards_received", 0)
            }
            reviews.append(review)
            
        logger.info(f"Fetched {len(reviews)} reviews for query: {query}")
        return reviews
        
    except Exception as e:
        logger.error(f"Error processing fetch_reviews results: {e}")
        return []

@mcp.tool()
def filter_genuine(reviews: List[Dict[str, Any]], min_length: int = 100, 
                  max_links: int = 2, min_score: int = 1) -> List[Dict[str, Any]]:
    """
    Advanced filtering for genuine reviews using multiple heuristics.
    
    Args:
        reviews: List of review objects
        min_length: Minimum character length for genuine reviews
        max_links: Maximum number of URLs allowed
        min_score: Minimum Reddit score/upvotes
    
    Returns:
        Filtered list of genuine review objects
    """
    filtered = []
    
    for review in reviews:
        text = review.get("text", "")
        score = review.get("score", 0)
        
        # Skip if too short
        if len(text) < min_length:
            continue
            
        # Skip if too many URLs/links
        if len(re.findall(r'https?://', text)) > max_links:
            continue
            
        # Skip if score too low
        if score < min_score:
            continue
            
        # Skip if suspicious repetition patterns
        word_counts = {}
        for word in re.findall(r'\w+', text.lower()):
            if len(word) > 3:  # Only check words longer than 3 chars
                word_counts[word] = word_counts.get(word, 0) + 1
                
        if word_counts and max(word_counts.values()) > 7:
            continue
            
        # Skip promotional content
        promotional_patterns = [
            r'discount code', r'promo code', r'use my link',
            r'sale now', r'click here', r'limited time offer',
            r'sponsored', r'\bAD\b', r'advertisement'
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in promotional_patterns):
            continue
            
        # Passed all filters
        filtered.append(review)
    
    logger.info(f"Filtered {len(filtered)} genuine reviews from {len(reviews)} total")
    return filtered

@mcp.tool()
def get_comments(post_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Fetch comments for a specific Reddit post.
    
    Args:
        post_id: The Reddit post ID
        limit: Maximum number of comments to return
        
    Returns:
        List of comments with text and metadata
    """
    url = REDDIT_POST_URL.format(post_id=post_id)
    params = {"limit": limit, "raw_json": 1}
    
    data = make_reddit_request(url, params)
    if "error" in data or not isinstance(data, list) or len(data) < 2:
        return []
        
    comments = []
    
    def extract_comments(comment_data, depth=0):
        if len(comments) >= limit:
            return
            
        if isinstance(comment_data, dict) and "data" in comment_data:
            if "body" in comment_data["data"]:
                comments.append({
                    "id": comment_data["data"].get("id", ""),
                    "text": comment_data["data"].get("body", ""),
                    "score": comment_data["data"].get("score", 0),
                    "author": comment_data["data"].get("author", ""),
                    "created_utc": comment_data["data"].get("created_utc", 0),
                    "depth": depth
                })
                
            # Process replies
            replies = comment_data["data"].get("replies", {})
            if isinstance(replies, dict) and "data" in replies:
                children = replies["data"].get("children", [])
                for child in children:
                    extract_comments(child, depth + 1)
    
    # Start with the top-level comments
    if len(data) > 1 and "data" in data[1] and "children" in data[1]["data"]:
        for comment in data[1]["data"]["children"]:
            extract_comments(comment)
    
    return comments[:limit]

@mcp.tool()
def analyze_sentiment(texts: List[str]) -> List[Dict[str, float]]:
    """
    Analyze sentiment of review texts.
    
    Uses either Cohere API if credentials available,
    or falls back to a basic lexicon-based approach.
    
    Args:
        texts: List of text strings to analyze
        
    Returns:
        List of sentiment scores with polarity and objectivity
    """
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    if cohere_api_key:
        # Use Cohere for advanced sentiment analysis
        try:
            headers = {
                "Authorization": f"Bearer {cohere_api_key}",
                "Content-Type": "application/json"
            }
            
            # Batch requests to avoid hitting limits
            batch_size = 10
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = requests.post(
                    "https://api.cohere.ai/v1/classify",
                    headers=headers,
                    json={
                        "inputs": batch,
                        "examples": [
                            {"text": "This product is amazing! Best purchase ever.", "label": "positive"},
                            {"text": "Works great and does exactly what it claims.", "label": "positive"},
                            {"text": "It's okay, nothing special but gets the job done.", "label": "neutral"},
                            {"text": "Basic functionality, not impressed but not disappointed.", "label": "neutral"},
                            {"text": "Terrible product, broke after a week. Waste of money.", "label": "negative"},
                            {"text": "Extremely disappointing, doesn't work as advertised.", "label": "negative"}
                        ]
                    }
                )
                
                data = response.json()
                if "classifications" in data:
                    for classification in data["classifications"]:
                        sentiment_score = 0
                        if classification["prediction"] == "positive":
                            sentiment_score = classification["confidence"]
                        elif classification["prediction"] == "negative":
                            sentiment_score = -classification["confidence"]
                            
                        results.append({
                            "polarity": sentiment_score,
                            "objectivity": 1 - max(classification["confidences"])
                        })
                
                # Be nice to the API
                time.sleep(1)
                
            return results
            
        except Exception as e:
            logger.error(f"Error with Cohere sentiment analysis: {e}")
            # Fall back to lexicon method
    
    # Basic lexicon-based sentiment analysis as fallback
    positive_words = set([
        'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic',
        'wonderful', 'best', 'love', 'perfect', 'recommend', 'happy',
        'pleased', 'impressive', 'outstanding', 'superb', 'superior',
        'exceptional', 'brilliant', 'satisfied', 'quality', 'well'
    ])
    
    negative_words = set([
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing',
        'waste', 'worst', 'hate', 'broken', 'defective', 'useless',
        'issue', 'problem', 'fail', 'cheap', 'avoid', 'return',
        'regret', 'disappointing', 'overpriced', 'difficult'
    ])
    
    intensifiers = set(['very', 'extremely', 'incredibly', 'absolutely', 'highly'])
    negations = set(['not', "don't", 'never', 'no', 'isn\'t', 'wasn\'t', 'aren\'t', 'doesn\'t'])
    
    results = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        pos_count = neg_count = 0
        word_count = len(words)
        
        # Skip empty texts
        if word_count == 0:
            results.append({"polarity": 0.0, "objectivity": 1.0})
            continue
            
        for i, word in enumerate(words):
            # Check for negation in the 3 preceding words
            negated = any(words[max(0, i-3):i].count(neg) for neg in negations)
            # Check for intensifiers in the 2 preceding words
            intensified = any(words[max(0, i-2):i].count(intens) for intens in intensifiers)
            
            multiplier = 1.5 if intensified else 1.0
            
            if word in positive_words:
                if negated:
                    neg_count += multiplier
                else:
                    pos_count += multiplier
            elif word in negative_words:
                if negated:
                    pos_count += multiplier
                else:
                    neg_count += multiplier
        
        # Calculate normalized sentiment score between -1 and 1
        denominator = max(1, pos_count + neg_count)
        polarity = (pos_count - neg_count) / denominator
        
        # Objectivity is higher when fewer sentiment words are found
        sentiment_word_ratio = denominator / max(1, word_count)
        objectivity = 1 - min(1, sentiment_word_ratio * 2)
        
        results.append({
            "polarity": polarity,
            "objectivity": objectivity
        })
        
    return results

@mcp.tool()
def extract_keywords(texts: List[str], max_keywords: int = 15) -> Dict[str, int]:
    """
    Extract significant keywords from review texts.
    
    Args:
        texts: List of review texts
        max_keywords: Maximum number of keywords to return
        
    Returns:
        Dictionary of keywords and their frequencies
    """
    # Common English stopwords
    stopwords = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
        't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 
        'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 
        'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 
        'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    ])
    
    # Additional words to ignore for product reviews
    product_stopwords = set([
        'product', 'item', 'buy', 'purchase', 'bought', 'ordered',
        'arrived', 'received', 'shipping', 'delivery', 'amazon',
        'review', 'star', 'stars', 'rating', 'recommend', 'recommendation',
        'price', 'cost', 'worth', 'value', 'money', 'paid', 'cheap', 'expensive'
    ])
    
    all_stopwords = stopwords.union(product_stopwords)
    
    # Combine all texts into one string
    combined_text = ' '.join(texts)
    
    # Extract all words (3+ chars) and convert to lowercase
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
    
    # Count word frequencies, excluding stopwords
    word_counts = {}
    for word in words:
        if word not in all_stopwords:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Find phrases (bigrams) that appear frequently
    bigrams = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        for i in range(len(words) - 1):
            if words[i] not in stopwords and words[i+1] not in stopwords:
                bigrams.append(f"{words[i]} {words[i+1]}")
    
    # Count bigram frequencies
    bigram_counts = {}
    for bigram in bigrams:
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    
    # Filter out low-frequency bigrams
    significant_bigrams = {b: c for b, c in bigram_counts.items() if c >= 2}
    
    # Combine single words and significant bigrams
    all_keywords = {**word_counts, **significant_bigrams}
    
    # Return top keywords by frequency
    sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_keywords[:max_keywords])

@mcp.tool()
def categorize_reviews(reviews: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize reviews based on content patterns.
    
    Args:
        reviews: List of review objects
        
    Returns:
        Dictionary with reviews grouped by category
    """
    categories = {
        "Usability": [],
        "Performance": [],
        "Value": [],
        "Quality": [],
        "Service": [],
        "Features": [],
        "Comparison": [],
        "Other": []
    }
    
    patterns = {
        "Usability": [
            r'easy to use', r'intuitive', r'user friendly', r'difficult to',
            r'learning curve', r'ergonomic', r'comfortable', r'interface',
            r'usability', r'controls', r'setup', r'instructions'
        ],
        "Performance": [
            r'performance', r'speed', r'fast', r'slow', r'responsive',
            r'efficient', r'powerful', r'weak', r'battery life', r'runtime',
            r'reliable', r'consistent', r'processing'
        ],
        "Value": [
            r'price', r'value', r'worth', r'expensive', r'cheap', r'affordable',
            r'overpriced', r'bargain', r'cost', r'money', r'investment',
            r'budget', r'return'
        ],
        "Quality": [
            r'quality', r'build', r'durable', r'sturdy', r'flimsy', r'broke',
            r'solid', r'material', r'construction', r'craftsmanship', r'design',
            r'premium', r'cheap feeling'
        ],
        "Service": [
            r'service', r'support', r'customer', r'warranty', r'return', r'help',
            r'assistance', r'responsive', r'team', r'replacement', r'refund',
            r'contact', r'phone', r'email', r'chat'
        ],
        "Features": [
            r'feature', r'functionality', r'capabilities', r'option', r'setting',
            r'mode', r'function', r'ability', r'does', r'doesn\'t', r'capable',
            r'versatile', r'flexible', r'adaptable'
        ],
        "Comparison": [
            r'better than', r'worse than', r'compared to', r'versus', r'vs',
            r'alternative', r'competitor', r'similar to', r'previous',
            r'upgraded from', r'switched from', r'instead of'
        ]
    }
    
    for review in reviews:
        text = review.get("text", "").lower()
        
        # Check each category's patterns
        categorized = False
        for category, pattern_list in patterns.items():
            if any(re.search(pattern, text) for pattern in pattern_list):
                categories[category].append(review)
                categorized = True
                break
                
        # Add to "Other" if no patterns matched
        if not categorized:
            categories["Other"].append(review)
    
    # Filter out empty categories
    return {cat: reviews for cat, reviews in categories.items() if reviews}

@mcp.tool()
def summarize_reviews(reviews: List[Dict[str, Any]], max_length: int = 500) -> str:
    """
    Generate a concise summary of review sentiment and key points.
    
    Args:
        reviews: List of review objects
        max_length: Maximum length of summary in characters
        
    Returns:
        Text summary of reviews
    """
    if not reviews:
        return "No reviews available to summarize."
    
    # Extract texts for sentiment analysis
    texts = [r.get("text", "") for r in reviews]
    
    # Get sentiments
    sentiments = analyze_sentiment(texts)
    
    # Calculate average sentiment
    avg_sentiment = sum(s.get("polarity", 0) for s in sentiments) / len(sentiments)
    
    # Count positive, neutral and negative reviews
    pos_count = sum(1 for s in sentiments if s.get("polarity", 0) > 0.2)
    neg_count = sum(1 for s in sentiments if s.get("polarity", 0) < -0.2)
    neu_count = len(sentiments) - pos_count - neg_count
    
    # Get keywords
    keywords = extract_keywords(texts, max_keywords=10)
    top_keywords = ", ".join(list(keywords.keys())[:5])
    
    # Get upvotes and comments info
    total_reviews = len(reviews)
    avg_score = sum(r.get("score", 0) for r in reviews) / max(1, total_reviews)
    
    # Build summary
    if avg_sentiment > 0.5:
        sentiment_desc = "extremely positive"
    elif avg_sentiment > 0.2:
        sentiment_desc = "generally positive"
    elif avg_sentiment > -0.2:
        sentiment_desc = "mixed or neutral"
    elif avg_sentiment > -0.5:
        sentiment_desc = "somewhat negative"
    else:
        sentiment_desc = "very negative"
    
    summary = (
        f"Analysis of {total_reviews} reviews shows {sentiment_desc} sentiment "
        f"({pos_count} positive, {neu_count} neutral, {neg_count} negative). "
        f"Reviews averaged {avg_score:.1f} upvotes. "
        f"Key topics mentioned: {top_keywords}."
    )
    
    # Include a few specific highlights if we have the space
    if len(summary) < (max_length - 100) and pos_count > 0:
        # Find highest-rated positive review
        pos_reviews = [r for i, r in enumerate(reviews) 
                    if i < len(sentiments) and sentiments[i].get("polarity", 0) > 0.2]
        if pos_reviews:
            top_pos = max(pos_reviews, key=lambda x: x.get("score", 0))
            pos_text = top_pos.get("text", "")
            pos_excerpt = pos_text[:70] + "..." if len(pos_text) > 70 else pos_text
            summary += f" Top positive highlight: \"{pos_excerpt}\""
    
    if len(summary) < (max_length - 100) and neg_count > 0:
        # Find highest-rated negative review
        neg_reviews = [r for i, r in enumerate(reviews) 
                    if i < len(sentiments) and sentiments[i].get("polarity", 0) < -0.2]
        if neg_reviews:
            top_neg = max(neg_reviews, key=lambda x: x.get("score", 0))
            neg_text = top_neg.get("text", "")
            neg_excerpt = neg_text[:70] + "..." if len(neg_text) > 70 else neg_text
            summary += f" Common criticism: \"{neg_excerpt}\""
    
    return summary[:max_length]

@mcp.tool()
def search_subreddits(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find relevant subreddits for a product or topic.
    
    Args:
        query: Search term
        limit: Maximum number of subreddits to return
        
    Returns:
        List of relevant subreddits with metadata
    """
    url = "https://www.reddit.com/subreddits/search.json"
    params = {
        "q": query,
        "limit": limit,
        "sort": "relevance"
    }
    
    data = make_reddit_request(url, params)
    
    if "error" in data:
        logger.error(f"Error in search_subreddits: {data['error']}")
        return []
    
    subreddits = []
    try:
        for child in data.get("data", {}).get("children", []):
            item = child.get("data", {})
            subreddit = {
                "name": item.get("display_name", ""),
                "title": item.get("title", ""),
                "description": item.get("public_description", ""),
                "subscribers": item.get("subscribers", 0),
                "url": f"https://reddit.com{item.get('url', '')}"
            }
            subreddits.append(subreddit)
        
        return subreddits
    except Exception as e:
        logger.error(f"Error processing subreddit search results: {e}")
        return []

@mcp.tool()
def compare_products(product1: str, product2: str, limit: int = 15) -> Dict[str, Any]:
    """
    Compare two products based on their reviews.
    
    Args:
        product1: First product name
        product2: Second product name
        limit: Maximum reviews to analyze per product
        
    Returns:
        Comparison data with reviews and analysis
    """
    # Fetch reviews for both products
    product1_reviews = fetch_reviews(product1, limit=limit)
    product2_reviews = fetch_reviews(product2, limit=limit)
    
    # Filter for genuine reviews
    genuine_p1 = filter_genuine(product1_reviews)
    genuine_p2 = filter_genuine(product2_reviews)
    
    # Extract texts
    texts_p1 = [r.get("text", "") for r in genuine_p1]
    texts_p2 = [r.get("text", "") for r in genuine_p2]
    
    # Analyze sentiment
    sentiment_p1 = analyze_sentiment(texts_p1)
    sentiment_p2 = analyze_sentiment(texts_p2)
    
    # Calculate average sentiment
    avg_sentiment_p1 = sum(s.get("polarity", 0) for s in sentiment_p1) / max(1, len(sentiment_p1))
    avg_sentiment_p2 = sum(s.get("polarity", 0) for s in sentiment_p2) / max(1, len(sentiment_p2))
    
    # Extract keywords
    keywords_p1 = extract_keywords(texts_p1, max_keywords=8)
    keywords_p2 = extract_keywords(texts_p2, max_keywords=8)
    
    # Build comparison data
    comparison = {
        "product1": {
            "name": product1,
            "review_count": len(genuine_p1),
            "avg_sentiment": avg_sentiment_p1,
            "keywords": list(keywords_p1.keys()),
            "reviews": genuine_p1[:3]  # Include sample reviews
        },
        "product2": {
            "name": product2,
            "review_count": len(genuine_p2),
            "avg_sentiment": avg_sentiment_p2,
            "keywords": list(keywords_p2.keys()),
            "reviews": genuine_p2[:3]  # Include sample reviews
        },
        "summary": f"Based on reviews, {product1} has {len(genuine_p1)} genuine reviews with "
                  f"a sentiment score of {avg_sentiment_p1:.2f}, while {product2} has "
                  f"{len(genuine_p2)} genuine reviews with a sentiment score of {avg_sentiment_p2:.2f}."
    }
    
    return comparison

@mcp.tool()
def deep_dive(query: str, max_posts: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a deep dive analysis on specific product aspects.
    
    Args:
        query: Product search query with optional aspect specification
        max_posts: Maximum number of posts to analyze deeply
        
    Returns:
        List of analysis results with posts and comments
    """
    # Parse query to separate product and aspect
    parts = query.split(" aspect:")
    product = parts[0].strip()
    aspect = parts[1].strip() if len(parts) > 1 else None
    
    # Refine search query based on aspect
    search_query = product
    if aspect:
        search_query = f"{product} {aspect}"
    
    # Get initial reviews
    reviews = fetch_reviews(search_query, limit=max_posts)
    genuine_reviews = filter_genuine(reviews)
    
    results = []
    for review in genuine_reviews[:max_posts]:
        # Extract post ID
        post_id = review.get("id", "")
        if not post_id and "permalink" in review:
            post_id = extract_post_id(review["permalink"])
            
        if post_id:
            # Get comments
            comments = get_comments(post_id, limit=10)
            
            # Calculate review sentiment
            review_sentiment = analyze_sentiment([review.get("text", "")])[0]
            
            # Analyze comment sentiments
            comment_texts = [c.get("text", "") for c in comments]
            comment_sentiments = analyze_sentiment(comment_texts)
            
            # Calculate average comment sentiment
            avg_comment_sentiment = (
                sum(s.get("polarity", 0) for s in comment_sentiments) / 
                max(1, len(comment_sentiments))
            )
            
            # Create result object
            result = {
                "review": review,
                "comments": comments,
                "review_sentiment": review_sentiment,
                "avg_comment_sentiment": avg_comment_sentiment,
                "aspect": aspect,
                "agreement": "high" if abs(review_sentiment.get("polarity", 0) - avg_comment_sentiment) < 0.3 else "low"
            }
            
            results.append(result)
    
    return results

# Run the server
if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", 8000))
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    logger.info(f"Starting MCP Reddit Review Analysis Server on {host}:{port}")
    mcp.host = host 
    mcp.port=port # Set host as an attribute
    mcp.run()
