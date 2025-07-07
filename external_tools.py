import os
import requests
from langchain.tools import Tool

# 1. Market Risk: Alpha Vantage
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
def get_alpha_vantage_quote(symbol: str) -> str:
    """Get the latest stock quote for a given symbol using Alpha Vantage."""
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "Global Quote" in data:
        quote = data["Global Quote"]
        return f"Price: {quote.get('05. price')}, Change: {quote.get('09. change')}, Volume: {quote.get('06. volume')}"
    return "No data found."
alpha_vantage_tool = Tool.from_function(
    func=get_alpha_vantage_quote,
    name="alpha_vantage_quote",
    description="Get latest stock quote and price change for a given symbol using Alpha Vantage."
)

# 3. Liquidity Risk: FRED
FRED_API_KEY = os.environ.get("FRED_API_KEY")
def get_fred_series(series_id: str) -> str:
    """Get latest value for a FRED economic data series (e.g., DFF, FEDFUNDS, IR, etc.)."""
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json", "sort_order": "desc", "limit": 1}
    response = requests.get(url, params=params)
    data = response.json()
    if "observations" in data and data["observations"]:
        obs = data["observations"][0]
        return f"{series_id}: {obs['value']} ({obs['date']})"
    return "No data found."
fred_tool = Tool.from_function(
    func=get_fred_series,
    name="fred_series",
    description="Get latest value for a FRED economic data series (interest rates, liquidity, etc.)."
)

# 4. Operational Risk: HaveIBeenPwned
HIBP_API_KEY = os.environ.get("HIBP_API_KEY")
def check_hibp(email: str) -> str:
    """Check if an email address has been in a data breach using HaveIBeenPwned."""
    url = f"https://haveibeenpwned.com/api/v3/breachedaccount/{email}"
    headers = {"hibp-api-key": HIBP_API_KEY, "user-agent": "risk-analyzer"}
    response = requests.get(url, headers=headers, params={"truncateResponse": "true"})
    if response.status_code == 200:
        return f"Breaches: {response.json()}"
    elif response.status_code == 404:
        return "No breaches found."
    return "Error or no data."
hibp_tool = Tool.from_function(
    func=check_hibp,
    name="hibp_breach_check",
    description="Check if an email address has been breached using HaveIBeenPwned."
)

# 6. General News: SerpAPI Google News
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
def get_google_news(query: str) -> str:
    """Get latest news headlines for a query using SerpAPI's Google News API."""
    url = "https://serpapi.com/search.json"
    params = {"q": query, "tbm": "nws", "api_key": SERPAPI_API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        news = response.json().get("news_results", [])
        if news:
            return " | ".join([n["title"] for n in news[:5]])
        return "No news found."
    return "Error or no data."
googlenews_tool = Tool.from_function(
    func=get_google_news,
    name="google_news",
    description="Get latest news headlines for a query using SerpAPI's Google News API."
)
