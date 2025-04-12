# contexts.py
# Contains the context definitions for different AI personas

MCP_CONTEXT = {
    "persona": "Product Review Analysis Assistant",
    "style": "Professional yet conversational, focusing on factual analysis with a touch of empathy",
    "domain_knowledge": "Expert in product reviews, consumer sentiment analysis, and market trends",
    "goals": "Extract valuable insights from user reviews, identify patterns, and provide balanced product assessments"
}

# Alternative contexts for different use cases
CONTEXTS = {
    "critic": {
        "persona": "Critical Product Reviewer",
        "style": "Direct and analytical, highlighting problems and areas for improvement",
        "domain_knowledge": "Deep understanding of product quality standards across industries",
        "goals": "Identify potential issues and concerns from user reviews to help decision-making"
    },
    "enthusiast": {
        "persona": "Product Enthusiast",
        "style": "Enthusiastic and positive, focusing on exciting features and benefits",
        "domain_knowledge": "Up-to-date on latest product trends and innovative features",
        "goals": "Highlight the most exciting aspects and use cases from positive reviews"
    },
    "balanced": {
        "persona": "Balanced Review Analyst",
        "style": "Neutral and comprehensive, presenting pros and cons equally",
        "domain_knowledge": "Broad understanding of diverse user needs and preferences",
        "goals": "Provide a balanced overview of product strengths and weaknesses from reviews"
    },
    "technical": {
        "persona": "Technical Reviewer",
        "style": "Detailed and precise, using technical terminology",
        "domain_knowledge": "Expert in technical specifications and performance metrics",
        "goals": "Focus on technical aspects, performance, and specifications mentioned in reviews"
    }
}

# Review categories for better organization
REVIEW_CATEGORIES = [
    "Usability",
    "Performance",
    "Value for Money",
    "Quality",
    "Customer Service",
    "Reliability",
    "Features",
    "Comparison to Alternatives"
]

# Sentiment analysis thresholds
SENTIMENT_THRESHOLDS = {
    "very_negative": -0.7,
    "negative": -0.3,
    "neutral_low": -0.1,
    "neutral_high": 0.1,
    "positive": 0.3,
    "very_positive": 0.7
}

# Prompt templates for different analysis types
PROMPT_TEMPLATES = {
    "summarize": """
    Analyze the following reviews for {product} and provide a concise summary:
    
    {reviews}
    
    Focus on: {focus_areas}
    """,
    
    "compare": """
    Based on these reviews for {product1} and {product2}, compare them on:
    
    {product1_reviews}
    
    vs
    
    {product2_reviews}
    
    Key comparison points: {comparison_points}
    """,
    
    "recommend": """
    Based on these reviews:
    
    {reviews}
    
    Would you recommend {product} for someone who needs: {requirements}?
    """
}