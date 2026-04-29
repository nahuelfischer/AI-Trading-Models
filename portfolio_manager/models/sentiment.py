"""
Sentiment module: scores financial news headlines via the Claude API.

Returns a float in [-1, +1]:
  +1 = strongly bullish
   0 = neutral
  -1 = strongly bearish

Designed to be called lazily (only when the portfolio engine requests a signal)
so it doesn't burn API calls during backtesting unless explicitly enabled.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import anthropic

from config.settings import SENTIMENT_MAX_TOKENS, SENTIMENT_MODEL

log = logging.getLogger(__name__)

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ─────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────

def score_headlines(
    ticker:    str,
    headlines: List[str],
    company:   str = "",
) -> float:
    """
    Send up to NEWS_MAX_HEADLINES headlines to Claude and get back a
    sentiment score in [-1, +1].  Returns 0.0 on failure.
    """
    if not headlines:
        return 0.0

    prompt = _build_prompt(ticker, headlines, company)

    try:
        client = _get_client()
        msg = client.messages.create(
            model=SENTIMENT_MODEL,
            max_tokens=SENTIMENT_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        return _parse_score(raw)
    except Exception as e:
        log.warning(f"[{ticker}] Sentiment API error: {e}")
        return 0.0


def score_tickers(
    ticker_headlines: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Convenience wrapper: score multiple tickers at once.
    Returns {ticker: score} dict.
    """
    return {
        ticker: score_headlines(ticker, headlines)
        for ticker, headlines in ticker_headlines.items()
    }


# ─────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────

def _build_prompt(ticker: str, headlines: List[str], company: str) -> str:
    lines = "\n".join(f"- {h}" for h in headlines)
    company_hint = f" ({company})" if company else ""
    return (
        f"You are a quantitative financial analyst. "
        f"Analyse the following news headlines for {ticker}{company_hint} "
        f"and return ONLY a JSON object with a single key 'score' whose value "
        f"is a float between -1.0 (extremely bearish) and +1.0 (extremely bullish). "
        f"0.0 is neutral. No explanation, no markdown, just JSON.\n\n"
        f"Headlines:\n{lines}\n\n"
        f"Respond only with: {{\"score\": <float>}}"
    )


def _parse_score(raw: str) -> float:
    try:
        data = json.loads(raw)
        score = float(data["score"])
        return max(-1.0, min(1.0, score))
    except (json.JSONDecodeError, KeyError, ValueError):
        # Try to extract a bare float if Claude didn't follow JSON format
        import re
        m = re.search(r"[-+]?\d*\.?\d+", raw)
        if m:
            return max(-1.0, min(1.0, float(m.group())))
        log.warning(f"Could not parse sentiment from: {raw!r}")
        return 0.0


# ─────────────────────────────────────────────
# News fetching (stub — plug in real NewsAPI / RSS)
# ─────────────────────────────────────────────

def fetch_headlines(ticker: str, max_items: int = 10) -> List[str]:
    """
    Fetch recent news headlines for a ticker.

    Currently returns a stub list.
    To use real headlines, replace this function with:
      - NewsAPI:  pip install newsapi-python
      - RSS feed: pip install feedparser
      - Alpha Vantage News endpoint
    """
    log.debug(f"[{ticker}] Using stub headlines (wire up a real news source here).")
    return [
        f"{ticker} reports strong quarterly earnings, beating consensus estimates.",
        f"Analysts raise price target for {ticker} following product launch.",
        f"Market uncertainty weighs on tech sector including {ticker}.",
    ]


# ─────────────────────────────────────────────
# Example integration with NewsAPI (commented out)
# ─────────────────────────────────────────────
# from newsapi import NewsApiClient
#
# def fetch_headlines_newsapi(ticker: str, max_items: int = 10) -> List[str]:
#     api = NewsApiClient(api_key=os.environ["NEWSAPI_KEY"])
#     resp = api.get_everything(q=ticker, language="en", sort_by="publishedAt", page_size=max_items)
#     return [a["title"] for a in resp.get("articles", [])][:max_items]
