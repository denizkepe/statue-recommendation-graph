import json
import re
import time
import signal
import sys
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Set, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# --- Configuration ---

BASE_URL = "https://karararama.yargitay.gov.tr"
DATA_DIR = Path("data/raw")
PROBE_DIR = DATA_DIR / "_probe"
LOG_DIR = Path("logs")

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROBE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup Logging
log_filename = LOG_DIR / f"scraper_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": BASE_URL,
    "Referer": BASE_URL + "/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

@dataclass
class ScrapeConfig:
    query: str = "iş hukuk"
    page_size: int = 50
    max_docs: int = 10000
    sleep_sec: float = 2.5  # slightly increased for politeness
    timeout: int = 45
    max_retries: int = 5

@dataclass
class ScraperState:
    last_page: int = 1
    total_collected_count: int = 0
    downloaded_ids: Set[str] = field(default_factory=set)
    skipped_pages: List[int] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "last_page": self.last_page,
            "total_collected_count": self.total_collected_count,
            "downloaded_ids": list(self.downloaded_ids),
            "skipped_pages": self.skipped_pages
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            last_page=data.get("last_page", 1),
            total_collected_count=data.get("total_collected_count", 0),
            downloaded_ids=set(data.get("downloaded_ids", [])),
            skipped_pages=data.get("skipped_pages", [])
        )

class YargitayScraper:
    def __init__(self, config: ScrapeConfig):
        self.cfg = config
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        
        self.run_tag = self._safe_filename(self.cfg.query)
        self.state_file = DATA_DIR / f"scraper_state_{self.run_tag}.json"
        
        self.state = self.load_state()
        self._setup_signals()

    def _safe_filename(self, s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s).strip("_")

    def _setup_signals(self):
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        logger.warning(f"\nExample Interrupt signal received ({signum}). Saving state and exiting...")
        self.save_state()
        sys.exit(0)

    def load_state(self) -> ScraperState:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                logger.info(f"Resuming from state file: {self.state_file}")
                return ScraperState.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load state file: {e}. Starting fresh.")
        return ScraperState()

    def save_state(self):
        try:
            temp_file = self.state_file.with_suffix('.tmp')
            temp_file.write_text(json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            temp_file.replace(self.state_file)
            logger.info("State saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def refresh_session(self):
        """Re-initialize the search session to fix timeouts/empty pages."""
        logger.info("Refreshing session with /arama endpoint...")
        try:
            # Random sleep before refresh to look natural
            time.sleep(2.0)
            self.post_json_retry(
                f"{BASE_URL}/arama", 
                {"data": {"aranan": self.cfg.query, "arananKelime": self.cfg.query}},
                is_refresh=True
            )
        except Exception as e:
            logger.error(f"Session refresh failed: {e}")

    def post_json_retry(self, url: str, payload: Dict[str, Any], is_refresh: bool = False) -> Any:
        # Don't infinite loop on refresh
        retries = 2 if is_refresh else self.cfg.max_retries
        
        for i in range(retries):
            try:
                resp = self.session.post(url, data=json.dumps(payload), timeout=self.cfg.timeout)
                resp.raise_for_status()
                try:
                    return resp.json()
                except ValueError:
                    return resp.text
            except Exception as e:
                wait_time = self.cfg.sleep_sec * (2 ** i) + 1 # Add base +1s
                logger.warning(f"Request failed (attempt {i+1}/{retries}): {e}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
        
        raise ConnectionError(f"Failed to fetch {url} after {retries} retries")

    def get_text_retry(self, url: str) -> str:
        for i in range(self.cfg.max_retries):
            try:
                resp = self.session.get(url, timeout=self.cfg.timeout)
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                wait_time = self.cfg.sleep_sec * (2 ** i) + 1
                logger.warning(f"Request failed (attempt {i+1}/{self.cfg.max_retries}): {e}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
        raise ConnectionError(f"Failed to fetch {url} after {self.cfg.max_retries} retries")

    def html_to_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n")
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def extract_items(self, response: Any) -> List[Dict[str, Any]]:
        # Same extraction logic as before, just cleaner
        if isinstance(response, dict):
            # Primary path
            try:
                items = response.get("data", {}).get("data", [])
                if isinstance(items, list):
                    return [x for x in items if isinstance(x, dict) and "id" in x]
            except Exception:
                pass
            
            # Fallback search
            for v in response.values():
                if isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, list) and vv and isinstance(vv[0], dict) and "id" in vv[0]:
                            return vv
                elif isinstance(v, list) and v and isinstance(v[0], dict) and "id" in v[0]:
                    return v
        return []

    def run(self):
        logger.info(f"Starting scrape for query: '{self.cfg.query}'")
        logger.info(f"Target: {self.cfg.max_docs} docs. Collected so far: {len(self.state.downloaded_ids)}")
        
        # Initial handshake
        self.refresh_session()

        pbar = tqdm(total=self.cfg.max_docs, initial=len(self.state.downloaded_ids), desc="Downloading", unit="doc")
        
        page = self.state.last_page
        consecutive_empty_pages = 0
        
        while len(self.state.downloaded_ids) < self.cfg.max_docs:
            if page in self.state.skipped_pages:
                logger.info(f"Skipping already marked problematic page {page}")
                page += 1
                continue

            payload = {
                "data": {
                    "aranan": self.cfg.query,
                    "arananKelime": self.cfg.query,
                    "pageSize": self.cfg.page_size,
                    "pageNumber": page
                }
            }

            logger.info(f"Fetching page {page}...")
            try:
                resp = self.post_json_retry(f"{BASE_URL}/aramalist", payload)
            except Exception:
                # If total failure, try one last session refresh then give up
                logger.warning("Fetch failed. Attempting session refresh and one last retry...")
                self.refresh_session()
                try:
                    resp = self.post_json_retry(f"{BASE_URL}/aramalist", payload)
                except Exception:
                    logger.error(f"Failed to fetch page {page} list after refresh. SKIPPING page.")
                    self.state.skipped_pages.append(page)
                    page += 1
                    self.save_state()
                    continue

            items = self.extract_items(resp)
            if not items:
                consecutive_empty_pages += 1
                logger.warning(f"No items found on page {page}. (Consecutive empty: {consecutive_empty_pages})")
                
                if consecutive_empty_pages >= 5:
                    logger.error(f"Consecutive empty pages threshold reached on page {page}. SKIPPING page to proceed.")
                    self.state.skipped_pages.append(page)
                    page += 1
                    # Don't reset skipped pages, keep track of them
                    # But DO reset consecutive empty counter because we are moving to a new page
                    # actually, we should maybe keep it high if the next page also fails? 
                    # Let's reset it to give the next page a fair chance (5 retries).
                    consecutive_empty_pages = 0 
                    self.save_state()
                    continue
                
                # Refresh session and RETRY SAME PAGE
                logger.info("Refreshing session and retrying same page...")
                self.refresh_session()
                time.sleep(5)
                continue
            
            # Reset counter if success
            consecutive_empty_pages = 0

            new_on_page = 0
            for item in items:
                if len(self.state.downloaded_ids) >= self.cfg.max_docs:
                    break
                    
                doc_id = str(item.get("id"))
                if not doc_id: 
                    continue
                
                # Skip if already downloaded
                if doc_id in self.state.downloaded_ids:
                    continue

                if (DATA_DIR / f"decision_{doc_id}.json").exists():
                     # Re-sync if file exists but not in state (e.g. state lost)
                    self.state.downloaded_ids.add(doc_id)
                    pbar.update(1)
                    continue

                # Download details
                try:
                    logger.debug(f"Downloading doc {doc_id}")
                    raw_html = self.get_text_retry(f"{BASE_URL}/getDokuman?id={doc_id}")
                    clean_text = self.html_to_text(raw_html)
                    
                    doc_data = {
                        "meta": {
                            "source": "yargitay",
                            "query": self.cfg.query,
                            "pageNumber": page,
                            "doc_id": doc_id,
                            "fetched_at": int(time.time()),
                            "original_meta": item
                        },
                        "raw_html": raw_html,
                        "text": clean_text
                    }
                    
                    with open(DATA_DIR / f"decision_{doc_id}.json", "w", encoding="utf-8") as f:
                        json.dump(doc_data, f, ensure_ascii=False, indent=2)
                    
                    self.state.downloaded_ids.add(doc_id)
                    pbar.update(1)
                    new_on_page += 1
                    
                    time.sleep(self.cfg.sleep_sec)
                    
                except Exception as e:
                    logger.error(f"Failed to download doc {doc_id}: {e}")
            
            # Update state after every page
            self.state.last_page = page + 1 # Next time start from next page
            self.save_state()
            
            # If we went through a whole page and found nothing new, but we haven't reached max docs,
            # it might mean we are scanning pages we already scraped in a previous run not tracked by state 
            # (though we handled that with the 'continue' checks). 
            # Just proceeding.
            page += 1

        pbar.close()
        self.save_state()
        logger.info(f"Scrape complete. Total docs: {len(self.state.downloaded_ids)}")
        if self.state.skipped_pages:
            logger.warning(f"Skipped {len(self.state.skipped_pages)} pages: {self.state.skipped_pages}")

def main():
    config = ScrapeConfig(timeout=60, max_retries=5)
    scraper = YargitayScraper(config)
    # Force start from page 60 as requested
    scraper.state.last_page = 60
    scraper.run()

if __name__ == "__main__":
    main()
