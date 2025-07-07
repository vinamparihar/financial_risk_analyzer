"""
This module defines the tools used by the financial analysis agents.

Each tool is a function that performs a specific action, such as searching the web or 
fetching financial data. These tools are then used by the agents to gather the information 
needed to perform their analysis.
"""

import os
import yfinance as yf
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from dotenv import load_dotenv
import pathlib

# Load environment variables from .env file in the project root
project_root = pathlib.Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# Set up the Tavily search tool
# This tool is used to perform web searches and gather news and articles.
def tavily_search_with_limit(query: str):
    rate_limiter.log_call("tavily")
    return TavilySearch(max_results=5)(query)

tavily_tool = Tool.from_function(
    func=tavily_search_with_limit,
    name="tavily_search",
    description=(
        "Web search for financial news and data. Use this tool to search for occurrences of risk-relevant keywords "
        "(e.g., 'Rate Hike', 'Credit Downgrade', 'Liquidity Crunch', 'Cyber Attack', 'Regulatory Fine') and parameters "
        "across market, credit, liquidity, operational, and regulatory risk categories. This supports sentiment-based risk scoring. "
        "Return findings that mention these keywords or parameters for UBS Group AG."
    )
)

# Set up the Yahoo Finance tool
# This tool is used to fetch financial data for a given stock ticker.
from langchain_core.tools import Tool

from financial_risk_analyzer.rate_limit import rate_limiter

def get_stock_info(ticker: str):
    """Fetches financial information for a given stock ticker from Yahoo Finance."""
    rate_limiter.log_call("yfinance")
    stock = yf.Ticker(ticker)
    # To reduce context, only return summary info
    info = stock.info
    summary = {k: info[k] for k in ('symbol','shortName','sector','industry','country','marketCap','regularMarketPrice') if k in info}
    return {
        "summary": summary,
        # Optionally, include only last 5 days of history
        "history": stock.history(period="5d").to_dict(),
    }

# Wrap get_stock_info as a LangChain tool
stock_info_tool = Tool.from_function(
    func=get_stock_info,
    name="get_stock_info",
    description=(
        "Get summarized stock info for a ticker. Use this tool to check for financial data related to risk-relevant parameters "
        "and keywords (e.g., yield curve, credit spread, liquidity ratio, operational incidents, regulatory compliance) for UBS Group AG. "
        "Supports the sentiment-based risk scoring system."
    )
)

# You can add more tools here as needed, for example, a tool to read a file
# or a tool to perform a specific calculation.
