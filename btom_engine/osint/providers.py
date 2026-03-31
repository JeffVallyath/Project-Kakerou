"""Provider interfaces for external data sources.

Providers are the lowest-level external interface. Adapters use them.
Mock providers enable fully offline testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchResult:
    """One search result from a provider."""
    title: str = ""
    url: str = ""
    snippet: str = ""


@dataclass
class PageContent:
    """Extracted content from a web page."""
    url: str = ""
    title: str = ""
    text: str = ""
    fetch_success: bool = False
    error: Optional[str] = None


class SearchProvider(ABC):
    """Abstract interface for search providers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        ...


class PageProvider(ABC):
    """Abstract interface for page content providers."""

    @abstractmethod
    def fetch(self, url: str, max_chars: int = 3000) -> PageContent:
        ...


# ---------------------------------------------------------------------------
# Mock providers for offline testing
# ---------------------------------------------------------------------------

class MockSearchProvider(SearchProvider):
    """Returns pre-loaded results for testing."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or []

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        return self._results[:top_k]


class MockPageProvider(PageProvider):
    """Returns pre-loaded page content for testing."""

    def __init__(self, pages: dict[str, PageContent] | None = None) -> None:
        self._pages = pages or {}

    def fetch(self, url: str, max_chars: int = 3000) -> PageContent:
        page = self._pages.get(url)
        if page:
            return PageContent(
                url=url,
                title=page.title,
                text=page.text[:max_chars],
                fetch_success=True,
            )
        return PageContent(url=url, fetch_success=False, error="URL not in mock data")


# ---------------------------------------------------------------------------
# Live providers (optional, feature-flagged)
# ---------------------------------------------------------------------------

class HttpPageProvider(PageProvider):
    """Fetches real web pages via httpx. Bounded text extraction."""

    def fetch(self, url: str, max_chars: int = 3000) -> PageContent:
        import httpx
        import re

        try:
            resp = httpx.get(url, timeout=15, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text

            title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""

            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            return PageContent(
                url=url,
                title=title,
                text=text[:max_chars],
                fetch_success=True,
            )

        except Exception as e:
            return PageContent(url=url, fetch_success=False, error=f"{type(e).__name__}: {e}")


class DuckDuckGoSearchProvider(SearchProvider):
    """Real search provider using DuckDuckGo HTML lite. No API key required.

    Bounded: top_k capped at 5. Timeout 10s. User-agent set to avoid blocks.
    Falls back safely on any error.
    """

    _DDG_URL = "https://lite.duckduckgo.com/lite/"
    _TIMEOUT = 10
    _MAX_RESULTS = 5

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        import httpx
        import re

        top_k = min(top_k, self._MAX_RESULTS)

        try:
            resp = httpx.post(
                self._DDG_URL,
                data={"q": query, "kl": ""},
                headers={"User-Agent": "Mozilla/5.0 (compatible; KakerouBot/1.0)"},
                timeout=self._TIMEOUT,
                follow_redirects=True,
            )
            resp.raise_for_status()
            html = resp.text

            results = self._parse_lite_html(html)
            return results[:top_k]

        except Exception as e:
            # Safe degradation — return empty results
            import logging
            logging.getLogger(__name__).warning("DuckDuckGo search failed: %s", e)
            return []

    @staticmethod
    def _parse_lite_html(html: str) -> list[SearchResult]:
        """Parse DuckDuckGo lite HTML results page."""
        import re

        results: list[SearchResult] = []

        # DDG lite uses a table-based layout with result links and snippets
        # Find all result links: <a rel="nofollow" href="URL" class="result-link">TITLE</a>
        link_pattern = re.compile(
            r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        # Find snippets in <td class="result-snippet">...</td>
        snippet_pattern = re.compile(
            r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
            re.DOTALL | re.IGNORECASE,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (url, title_html) in enumerate(links):
            # Strip HTML from title
            title = re.sub(r"<[^>]+>", "", title_html).strip()
            # Get corresponding snippet if available
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

            if url and url.startswith("http"):
                results.append(SearchResult(
                    title=title[:200],
                    url=url,
                    snippet=snippet[:300],
                ))

        # Fallback: if the lite parser didn't find structured results,
        # try a simpler pattern for any external links
        if not results:
            simple_links = re.findall(r'href="(https?://[^"]+)"[^>]*>([^<]+)</a>', html)
            for url, title in simple_links[:5]:
                if "duckduckgo.com" not in url:
                    results.append(SearchResult(
                        title=title.strip()[:200],
                        url=url,
                        snippet="",
                    ))

        return results
