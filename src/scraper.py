# src/scrape_shl.py  (improved version)
import requests
from bs4 import BeautifulSoup
import json
import time
from tqdm import tqdm
import os
from urllib.parse import urljoin

# Optional: add your logger here
# from src.logger import logger
# For now using print; replace with logger.info / logger.error later

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
}

def get_page_items(start=0):
    url = f"{BASE_URL}?type=1&start={start}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        print(f"Fetched page successfully: {url} (status {resp.status_code})")
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Find all tables
    tables = soup.find_all("table")
    print(f"Found {len(tables)} <table> elements on the page")
    
    if len(tables) == 0:
        print("No tables at all — site may be broken or JS-heavy now. Dumping first 2000 chars of HTML:")
        print(soup.prettify()[:2000])
        return []

    target_table = None
    for table in tables:
        # Look for header row or th containing "Individual Test Solutions"
        header_row = table.find("tr")
        if header_row:
            header_text = header_row.get_text(strip=True).lower()
            if "individual test solutions" in header_text:
                target_table = table
                print("Found Individual Test Solutions table via header text match")
                break

    if not target_table:
        # Fallback: take the second table if header search fails
        if len(tables) >= 2:
            target_table = tables[1]
            print("Header text not found — falling back to second table")
        else:
            print("No suitable table found even with fallback. Dumping soup:")
            print(soup.prettify()[:1500])
            return []

    rows = target_table.find_all("tr")[1:]  # skip header
    print(f"Extracted {len(rows)} rows from target table")

    items = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        name_col = cols[0]
        link_tag = name_col.find("a", href=True)
        if not link_tag:
            continue

        name = link_tag.get_text(strip=True)
        if not name:
            continue

        relative_url = link_tag["href"]
        full_url = urljoin(BASE_URL, relative_url)

        test_type_str = cols[3].get_text(strip=True)
        test_types = test_type_str.split() if test_type_str else []

        items.append({
            "name": name,
            "url": full_url,
            "test_types": test_types,
            "raw_list_page_data": {
                "remote": cols[1].get_text(strip=True),
                "adaptive": cols[2].get_text(strip=True),
            }
        })

    return items
def enrich_detail_page(item):
    try:
        resp = requests.get(item["url"], headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"Detail page failed {item['url']}: {e}")
        return item

    soup = BeautifulSoup(resp.text, "html.parser")

    # Description: try more flexible selectors
    desc_container = (
        soup.find("div", class_="product-description")
        or soup.find("div", class_="rich-text")
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", id="content")
    )
    description = (
        desc_container.get_text(separator=" ", strip=True)[:2200]
        if desc_container
        else "No description found"
    )

    # Duration: improved pattern matching
    duration = 0
    text_all = soup.get_text().lower()
    import re
    match = re.search(r'duration\D*?(\d+)\s*(?:minute|min)', text_all)
    if match:
        duration = int(match.group(1))

    # Adaptive / Remote
    adaptive = "Yes" if any(w in text_all for w in ["adaptive", "irt", "computer adaptive"]) else "No"
    remote   = "Yes" if any(w in text_all for w in ["remote", "untimed", "webcam"]) else "No"

    item.update({
        "description": description,
        "duration_minutes": duration,
        "adaptive_support": adaptive,
        "remote_support": remote,
    })
    return item


# ────────────────────────────────────────────────
# MAIN CRAWL
# ────────────────────────────────────────────────

all_items = []
MAX_PAGES = 32  # safety

for page in tqdm(range(MAX_PAGES), desc="Crawling pages"):
    start = page * 12
    page_items = get_page_items(start)
    if not page_items:
        print(f"Stopping at page {page} — no more items")
        break

    for item in tqdm(page_items, desc=f"Detail pages {page+1}", leave=False):
        enrich_detail_page(item)
        all_items.append(item)
        time.sleep(1.4)  # polite delay

    time.sleep(3)  # between pages

print(f"\nFinished. Total items crawled: {len(all_items)}")

os.makedirs("data", exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M")
filename = f"data/shl_individual_tests_{timestamp}.json"

with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_items, f, ensure_ascii=False, indent=2)

print(f"Saved to: {filename}")
print("Next step: Check if len >= 377. If low → open browser dev tools on page and adjust selectors.")