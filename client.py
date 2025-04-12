import os
import sys
import asyncio
import argparse
import json
import time
from typing import Dict, List, Any, Optional, Union
from tabulate import tabulate
import requests
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from contexts import MCP_CONTEXT, CONTEXTS, REVIEW_CATEGORIES, SENTIMENT_THRESHOLDS, PROMPT_TEMPLATES

# Load environment variables
load_dotenv()

# Constants and configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "mixtral-8x7b-32768"  # Can be configured based on needs

class MCPRedditAgent:
    """
    Advanced Reddit Review Analysis Agent using MCP and Groq
    """
    def __init__(self, server_path: str):
        self.server_path = server_path
        self.session = None
        self.stdio_r = None
        self.stdio_w = None
        self.available_tools = []
        self.context = MCP_CONTEXT
        self.results_cache = {}

    async def connect(self):
        """Connect to the MCP server"""
        params = StdioServerParameters(command="python", args=[self.server_path], env=None)
        
        try:
            # Use async with for context manager
            async with stdio_client(params) as (self.stdio_r, self.stdio_w):
                self.session = ClientSession(self.stdio_r, self.stdio_w)
                await self.session.initialize()
                tools_response = await self.session.list_tools()
                self.available_tools = tools_response.tools
                print(f"🛠️ Connected to server. Available tools: {[t.name for t in self.available_tools]}")
                return True
        except Exception as e:
            print(f"❌ Error connecting to server: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session:
            await self.session.close()
        if self.stdio_w:
            self.stdio_w.close()

    def call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call the Groq LLM API"""
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY not set in environment variables"
            
        try:
            resp = requests.post(
                GROQ_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "system", "content": prompt}],
                    "temperature": temperature
                },
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            print(f"❌ Error calling Groq API: {e}")
            return f"Error connecting to LLM service: {str(e)}"

    def build_prompt(self, context: dict, data: Any, question: str) -> str:
        """Build a prompt for the LLM"""
        header = (
            f"You are a {context['persona']}.\n"
            f"Style: {context['style']}\n"
            f"Knowledge: {context['domain_knowledge']}\n"
            f"Goal: {context['goals']}\n\n"
        )
        
        data_section = ""
        if isinstance(data, list) and len(data) > 0:
            # Format review data
            data_section = "Here are some relevant user reviews:\n\n"
            for i, review in enumerate(data[:5], 1):
                if isinstance(review, dict):
                    text = review.get("text", "")
                    score = review.get("score", 0)
                    subreddit = review.get("subreddit", "")
                    data_section += f"Review #{i} [{score} upvotes from r/{subreddit}]:\n{text[:300]}...\n\n"
                else:
                    data_section += f"Review #{i}: {str(review)[:300]}...\n\n"
        elif isinstance(data, dict):
            # Format other structured data
            data_section = "Here is the relevant data:\n\n"
            data_section += json.dumps(data, indent=2)
        else:
            data_section = str(data)
            
        return (
            header +
            f"User question: {question}\n\n" +
            data_section + "\n\n" +
            "Based on the above, provide a comprehensive yet concise answer. " +
            "Focus on insights rather than just summarizing the reviews."
        )

    async def change_persona(self, persona_name: str):
        """Change the agent's persona"""
        if persona_name in CONTEXTS:
            self.context = CONTEXTS[persona_name]
            return f"Persona changed to: {persona_name}"
        else:
            return f"Persona not found. Available personas: {list(CONTEXTS.keys())}"

    async def fetch_and_analyze(self, query: str, limit: int = 30, 
                               timeframe: str = "month") -> Dict[str, Any]:
        """Fetch, filter, and analyze reviews"""
        # 1. Fetch raw reviews
        print(f"🔍 Fetching reviews for '{query}'...")
        raw_reviews = await self.session.call_tool("fetch_reviews", {
            "query": query, 
            "limit": limit,
            "timeframe": timeframe
        })
        reviews = raw_reviews.content
        print(f"📚 Fetched {len(reviews)} raw reviews.")
        
        if not reviews:
            return {"error": "No reviews found"}
            
        # 2. Filter genuine reviews
        print("🧐 Filtering genuine reviews...")
        genuine_result = await self.session.call_tool("filter_genuine", {"reviews": reviews})
        genuine_reviews = genuine_result.content
        print(f"✅ {len(genuine_reviews)} reviews passed genuine filter.")
        
        if not genuine_reviews:
            return {"error": "No genuine reviews found"}
            
        # 3. Get review texts
        texts = [r.get("text", "") for r in genuine_reviews]
        
        # 4. Analyze sentiment
        print("📊 Analyzing sentiment...")
        sentiment_result = await self.session.call_tool("analyze_sentiment", {"texts": texts})
        sentiments = sentiment_result.content
        
        # 5. Extract keywords
        print("🔑 Extracting keywords...")
        keywords_result = await self.session.call_tool("extract_keywords", {"texts": texts})
        keywords = keywords_result.content
        
        # 6. Categorize reviews
        print("📋 Categorizing reviews...")
        categories_result = await self.session.call_tool("categorize_reviews", {"reviews": genuine_reviews})
        categories = categories_result.content
        
        # 7. Generate summary
        print("📝 Generating summary...")
        summary_result = await self.session.call_tool("summarize_reviews", {"reviews": genuine_reviews})
        summary = summary_result.content
        
        # 8. Compile results
        results = {
            "query": query,
            "total_reviews": len(reviews),
            "genuine_reviews": len(genuine_reviews),
            "reviews": genuine_reviews,
            "sentiments": sentiments,
            "keywords": keywords,
            "categories": categories,
            "summary": summary
        }
        
        # Cache results
        self.results_cache[query] = results
        return results

    async def answer_question(self, query: str, question: str) -> str:
        """Answer a specific question about reviews"""
        # Check if we have cached results
        results = self.results_cache.get(query)
        
        if not results:
            print(f"No cached data for '{query}'. Fetching new data...")
            results = await self.fetch_and_analyze(query)
            
        if "error" in results:
            return f"Error: {results['error']}"
            
        # Build prompt for the LLM
        prompt = self.build_prompt(self.context, results["reviews"], question)
        
        # Call the LLM
        print("🤖 Consulting LLM for answer...")
        answer = self.call_llm(prompt)
        
        return answer

    async def compare_products(self, product1: str, product2: str) -> str:
        """Compare two products based on their reviews"""
        print(f"🔄 Comparing '{product1}' and '{product2}'...")
        
        comparison_result = await self.session.call_tool("compare_products", {
            "product1": product1,
            "product2": product2,
            "limit": 15
        })
        comparison = comparison_result.content
        
        # Format the comparison data for the LLM
        prompt = (
            f"You are a {self.context['persona']}.\n"
            f"Style: {self.context['style']}\n"
            f"Knowledge: {self.context['domain_knowledge']}\n"
            f"Goal: Compare these two products fairly based on user reviews.\n\n"
            "Product comparison data:\n"
            f"{json.dumps(comparison, indent=2)}\n\n"
            "Please provide a detailed comparison of these products based on the review data. "
            "Highlight key differences, strengths, and weaknesses of each. "
            "Focus on insights rather than just raw data."
        )
        
        print("🤖 Generating comparison analysis...")
        analysis = self.call_llm(prompt)
        
        return analysis

    async def deep_review_analysis(self, query: str, aspect: Optional[str] = None) -> str:
        """Perform a deep dive analysis on specific product reviews"""
        search_query = query
        if aspect:
            search_query = f"{query} aspect:{aspect}"
        
        print(f"🔍 Deep diving into '{search_query}'...")
        dive_result = await self.session.call_tool("deep_dive", {
            "query": search_query,
            "max_posts": 5
        })
        dive_data = dive_result.content
        
        # Format the data for the LLM
        prompt = (
            f"You are a {self.context['persona']}.\n"
            f"Style: {self.context['style']}\n"
            f"Knowledge: {self.context['domain_knowledge']}\n"
            f"Goal: Provide deep insights on specific aspects of product reviews.\n\n"
            f"Deep analysis data for '{query}'"
            f"{' focusing on aspect: ' + aspect if aspect else ''}:\n"
            f"{json.dumps(dive_data, indent=2)}\n\n"
            "Please provide an insightful analysis of these reviews and comments. "
            "Identify patterns, notable feedback, and key insights. "
            "Consider both the original posts and the comments/reactions."
        )
        
        print("🤖 Generating deep analysis...")
        analysis = self.call_llm(prompt)
        
        return analysis

    def print_results_table(self, results):
        """Print results in a nicely formatted table"""
        if not results or "error" in results:
            print(f"❌ Error: {results.get('error', 'Unknown error')}")
            return
            
        # Print summary
        print("\n" + "="*80)
        print(f"📊 ANALYSIS RESULTS FOR: {results['query']}")
        print("="*80)
        print(f"📝 SUMMARY: {results['summary']}")
        print("-"*80)
        
        # Print keyword table
        print("\n🔑 TOP KEYWORDS:")
        keywords_list = list(results['keywords'].items())
        if keywords_list:
            keywords_table = [[kw, count] for kw, count in keywords_list[:10]]
            print(tabulate(keywords_table, headers=["Keyword", "Count"], tablefmt="plain"))
        
        # Print sentiment distribution
        print("\n📊 SENTIMENT DISTRIBUTION:")
        sentiments = results.get('sentiments', [])
        pos = sum(1 for s in sentiments if s.get('polarity', 0) > 0.2)
        neg = sum(1 for s in sentiments if s.get('polarity', 0) < -0.2)
        neu = len(sentiments) - pos - neg
        
        sentiment_table = [
            ["Positive", pos, f"{pos/max(1, len(sentiments))*100:.1f}%"],
            ["Neutral", neu, f"{neu/max(1, len(sentiments))*100:.1f}%"],
            ["Negative", neg, f"{neg/max(1, len(sentiments))*100:.1f}%"]
        ]
        print(tabulate(sentiment_table, headers=["Sentiment", "Count", "Percentage"], tablefmt="plain"))
        
        # Print categories
        print("\n📋 REVIEW CATEGORIES:")
        categories = results.get('categories', {})
        if categories:
            cat_table = [[cat, len(revs)] for cat, revs in categories.items()]
            print(tabulate(cat_table, headers=["Category", "Count"], tablefmt="plain"))
        
        # Print sample reviews
        print("\n📚 SAMPLE REVIEWS:")
        reviews = results.get('reviews', [])
        for i, review in enumerate(reviews[:3], 1):
            print(f"\nReview #{i} - Score: {review.get('score', 'N/A')} - Subreddit: r/{review.get('subreddit', 'N/A')}")
            text = review.get('text', '')
            print(f"{text[:200]}..." if len(text) > 200 else text)
            
        print("\n" + "="*80)

async def main():
    parser = argparse.ArgumentParser(description="MCP Reddit Review Analysis Agent")
    parser.add_argument("server_path", help="Path to server.py script")
    parser.add_argument("--mode", "-m", choices=["interactive", "analyze", "compare", "deep"], 
                       default="interactive", help="Operation mode")
    parser.add_argument("--query", "-q", help="Search query for product/topic")
    parser.add_argument("--question", help="Specific question about the product")
    parser.add_argument("--product1", help="First product for comparison")
    parser.add_argument("--product2", help="Second product for comparison")
    parser.add_argument("--aspect", help="Specific aspect for deep analysis")
    parser.add_argument("--persona", "-p", choices=list(CONTEXTS.keys()),
                       help="Set agent persona")
    parser.add_argument("--limit", "-l", type=int, default=30,
                       help="Maximum number of reviews to fetch")
    
    args = parser.parse_args()

    agent = MCPRedditAgent(args.server_path)
    
    try:
        if not await agent.connect():
            print("Failed to connect to server. Exiting.")
            return 1
        
        if args.persona:
            result = await agent.change_persona(args.persona)
            print(result)
        
        if args.mode == "interactive":
            await interactive_mode(agent)
        elif args.mode == "analyze":
            if not args.query:
                print("Error: --query is required for analyze mode")
                return 1
                
            results = await agent.fetch_and_analyze(args.query, args.limit)
            agent.print_results_table(results)
            
            if args.question:
                answer = await agent.answer_question(args.query, args.question)
                print("\n❓ QUESTION:", args.question)
                print("📝 ANSWER:", answer)
        elif args.mode == "compare":
            if not args.product1 or not args.product2:
                print("Error: --product1 and --product2 are required for compare mode")
                return 1
                
            analysis = await agent.compare_products(args.product1, args.product2)
            print("\n🔄 COMPARISON ANALYSIS:")
            print(analysis)
        elif args.mode == "deep":
            if not args.query:
                print("Error: --query is required for deep mode")
                return 1
                
            analysis = await agent.deep_review_analysis(args.query, args.aspect)
            print("\n🔍 DEEP ANALYSIS:")
            print(analysis)
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
    finally:
        await agent.disconnect()
    
    return 0

async def interactive_mode(agent: MCPRedditAgent):
    """Interactive command-line interface for the agent"""
    print("\n" + "="*80)
    print("🤖 Reddit Review Analysis Agent - Interactive Mode")
    print("="*80)
    print("Type 'help' to see available commands, 'exit' to quit")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ('exit', 'quit'):
                break
                
            elif command.lower() == 'help':
                print("\nAvailable commands:")
                print("  analyze <product/topic> [limit]    - Analyze reviews for a product")
                print("  ask <product/topic> <question>     - Ask a question about a product")
                print("  compare <product1> <product2>      - Compare two products")
                print("  deep <product> [aspect]            - Deep dive analysis on a product")
                print("  persona <name>                     - Change agent persona")
                print("  personas                           - List available personas")
                print("  exit                               - Exit the program")
                
            elif command.lower() == 'personas':
                print("\nAvailable personas:")
                for name, context in CONTEXTS.items():
                    print(f"  {name}: {context['persona']} - {context['goals']}")
                
            elif command.lower().startswith('persona '):
                persona = command[8:].strip()
                result = await agent.change_persona(persona)
                print(result)
                
            elif command.lower().startswith('analyze '):
                parts = command[8:].split()
                query = parts[0]
                limit = int(parts[1]) if len(parts) > 1 else 30
                
                print(f"Analyzing '{query}' with limit {limit}...")
                results = await agent.fetch_and_analyze(query, limit)
                agent.print_results_table(results)
                
            elif command.lower().startswith('ask '):
                parts = command[4:].split(' ', 1)
                if len(parts) < 2:
                    print("Usage: ask <product/topic> <question>")
                    continue
                    
                query = parts[0]
                question = parts[1]
                
                answer = await agent.answer_question(query, question)
                print("\n❓ QUESTION:", question)
                print("📝 ANSWER:", answer)
                
            elif command.lower().startswith('compare '):
                parts = command[8:].split(' ', 1)
                if len(parts) < 2:
                    print("Usage: compare <product1> <product2>")
                    continue
                    
                product1 = parts[0]
                product2 = parts[1]
                
                analysis = await agent.compare_products(product1, product2)
                print("\n🔄 COMPARISON ANALYSIS:")
                print(analysis)
                
            elif command.lower().startswith('deep '):
                parts = command[5:].split(' ', 1)
                query = parts[0]
                aspect = parts[1] if len(parts) > 1 else None
                
                analysis = await agent.deep_review_analysis(query, aspect)
                print("\n🔍 DEEP ANALYSIS:")
                print(analysis)
                
            else:
                print("Unknown command. Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\nCommand interrupted.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))