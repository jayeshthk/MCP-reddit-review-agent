# Reddit Review Analysis Agent

A powerfull tool for mining insites from Reddit product reviews MCP(model context protocol).

## What does it do?

This agent helps you discover what Reddit users are saying about products, by:

- Finding & filtering genuine product reviews across subreddits
- Analysing sentiment and key topics
- Summarizing review highlights
- Comparing products side-by-side
- Answering specific questions about products
- Deep diving into specific product aspects

## Features

- **Advanced Reddit Scraping**: Searches across multiple subreddits for most relevant discussons
- **'Smart' Filtering**: Removes spam, fake reviews, and irrelevent content
- **Sentiment Analysis**: Understand if reviews are positive, negative or mixed
- **Keyword Extraction**: Identify most common topics in reviews
- **Review Categorization**: Group reviews by features, bugs, price, etc.
- **Product Comparison**: See how two products stack up against each other
- **Deep-Dive Analysis**: Explore specific aspects of products (battery, performance, etc)
- **Customisable Personas**: Switch between different analyst "personalities"

## Installation

1. Clone this repository

```bash
git clone https://github.com/jayeshthk/MCP-reddit-review-agent.git
cd MCP-reddit-review-agent
```

2. Install dependancies

```bash
pip install -r requirements.txt
```

3. Create an `.env` file with your API keys:

```
# API Keys
COHERE_API_KEY=your_cohere_api_key_here
GROQ_API_KEY=your_groq_api_key
# Reddit API Credentials (for authenticated access)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=MCP-Reddit-Review-Agent/1.0

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

## Usage Examples

### Interactive Mode

```bash
python client.py server.py --mode interactive
```

This opens an interactive shell where you can type various commands:

```
> analyze iPhone 15
> ask iPhone 15 How is the battery life?
> compare iPhone 15 Samsung Galaxy S24
> deep MacBook Pro performance
```

### Command-line Mode

Direct analysis:

```bash
python client.py server.py --mode analyze --query "Sony WH-1000XM5" --question "How is the battery life?"
```

Product comparison:

```bash
python client.py server.py --mode compare --product1 "iPhone 15" --product2 "Samsung Galaxy S24"
```

Deep analysis:

```bash
python client.py server.py --mode deep --query "MacBook Pro" --aspect "performance"
```

Change persona:

```bash
python client.py server.py --mode analyze --query "Steam Deck" --persona tech_critic
```

## Available Commands

In interactive mode, the following commands are available:

- `analyze <product/topic> [limit]` - Analyze reviews for a product
- `ask <product/topic> <question>` - Ask a question about a product
- `compare <product1> <product2>` - Compare two products
- `deep <product> [aspect]` - Deep dive analysis on a product
- `persona <name>` - Change agent persona
- `personas` - List available personas
- `exit` - Exit the program

## Notes & Limitations

- Reddit's API has rate limits. If you encounter errors, try again later.
- The qualitiy of analysis depends on available Reddit discussions.
- Very new or obscure products may have limited data available.
- The tool is designed for product reviews and may not work well for other topics.
- Running some queries can take 30-60 seconds depending on the amount of data.

## Troubleshooting

**Issue**: `Error connecting to server`

- Make sure server.py is running in a separate terminal

**Issue**: `Rate limit exceeded`

- Reddit API enforces rate limits - wait 5-10 minutes before trying again

**Issue**: `No genuine reviews found`

- Try broadening your search query or checking spelling
- Some products might just not have many Reddit discussions

## Requirements 📦

- Python 3.8+
- MCP library
- Reddit API credentials
- Groq API key (for LLM analysis)

## Future plans

- Add multi-platform support (Twitter, Amazon reviews, etc)
- Implement visual charts and graphs for results
- Create a web UI for easier interaction
- Add export options (CSV, PDF reports)
