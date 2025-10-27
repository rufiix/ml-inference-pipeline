import json
import logging
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BASE_URL = "https://scikit-learn.org/stable/"
USER_GUIDE_URL = urljoin(BASE_URL, "user_guide.html")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "scikit-learn_docs.jsonl"
HEADERS = {
    "User-Agent": "NexusQuery Scraper/1.0"
}

def get_page_content(url: str) -> str:
    """Fetches the HTML content of a given URL."""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return ""

def get_user_guide_links(html_content: str) -> list[str]:
    """Extracts all links to individual user guide pages."""
    soup = BeautifulSoup(html_content, "html.parser")
    links = set()
    # Find all list items within the specific section that contains user guide topics
    guide_list = soup.select("li.toctree-l1 > a.reference.internal")
    for link in guide_list:
        href = link.get("href")
        if href and not href.startswith("http"):
            full_url = urljoin(BASE_URL, href)
            links.add(full_url)
    logging.info(f"Found {len(links)} unique user guide links.")
    return list(links)

def extract_main_content(url: str, html_content: str) -> dict | None:
    """Extracts the main textual content from a documentation page."""
    soup = BeautifulSoup(html_content, "html.parser")
    main_content = soup.select_one("main[role='main']")

    if not main_content:
        logging.warning(f"No main content found for {url}")
        return None

    # Remove code blocks, navigation, and other non-essential elements
    for element in main_content.select(".sphx-glr-thumbcontainer, .btn, .breadcrumb-nav"):
        element.decompose()

    text = main_content.get_text(separator="\n", strip=True)

    return {
        "source_url": url,
        "text": text
    }

def main():
    """Main function to scrape documentation and save it."""
    logging.info("Starting Scikit-learn documentation scraping process.")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Get the main user guide page to find all sub-links
    logging.info(f"Fetching user guide index from {USER_GUIDE_URL}...")
    main_page_html = get_page_content(USER_GUIDE_URL)
    if not main_page_html:
        logging.critical("Could not fetch the main user guide page. Exiting.")
        return

    links = get_user_guide_links(main_page_html)

    if not links:
        logging.critical("No user guide links found. Exiting.")
        return

    # 2. Scrape each link and save the content
    scraped_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, link in enumerate(links):
            logging.info(f"[{i+1}/{len(links)}] Scraping {link}...")
            page_html = get_page_content(link)
            if page_html:
                content = extract_main_content(link, page_html)
                if content:
                    f.write(json.dumps(content) + "\n")
                    scraped_count += 1
            # Be a good web citizen
            time.sleep(0.1)

    logging.info(f"Scraping complete. Successfully saved content from {scraped_count}/{len(links)} pages to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
