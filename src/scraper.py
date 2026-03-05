# src/scraper.py
import requests
from bs4 import BeautifulSoup
import json
import time
from tqdm import tqdm
import os
from urllib.parse import urljoin
import re
from pathlib import Path
from config import SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX, MAX_PAGES, logger

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
}

def get_page_items(start: int = 0):
    url = f"{BASE_URL}?type=1&start={start}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        logger.info(f"Fetched page: {url}  (status {resp.status_code})")
    except Exception as e:
        logger.error(f"Page fetch failed {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        logger.warning("No <table> elements found. Dumping head of HTML.")
        logger.debug(soup.prettify()[:2000])
        return []

    target_table = None
    for table in tables:
        header_row = table.find("tr")
        if header_row and "individual test solutions" in header_row.get_text(strip=True).lower():
            target_table = table
            logger.info("Found Individual Test Solutions table")
            break

    if not target_table and len(tables) >= 2:
        target_table = tables[1]
        logger.info("Using fallback: second table")

    if not target_table:
        logger.error("No suitable table found.")
        return []

    rows = target_table.find_all("tr")[1:]
    logger.info(f"Extracted {len(rows)} candidate rows")

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

        full_url = urljoin(BASE_URL, link_tag["href"])

        test_type_str = cols[3].get_text(strip=True)
        test_types = [t.strip() for t in test_type_str] if test_type_str else []  # split into individual letters
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


def enrich_detail_page(item: dict) -> dict:
    try:
        resp = requests.get(item["url"], headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Detail page failed {item['url']}: {e}")
        return item

    soup = BeautifulSoup(resp.text, "html.parser")

    desc_container = (
        soup.find("div", class_="product-description")
        or soup.find("div", class_="rich-text")
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", id="content")
    )
    description = desc_container.get_text(separator=" ", strip=True)[:2200] if desc_container else "No description"

    text_all = soup.get_text().lower()
    duration = 0
    m = re.search(r'duration\D*?(\d+)\s*(?:minute|min|m)', text_all)
    if m:
        duration = int(m.group(1))

    adaptive = "Yes" if any(w in text_all for w in ["adaptive", "irt", "computer adaptive"]) else "No"
    remote   = "Yes" if any(w in text_all for w in ["remote", "untimed", "webcam"]) else "No"

    item.update({
        "description": description,
        "duration_minutes": duration,
        "adaptive_support": adaptive,
        "remote_support": remote,
    })
    return item


def run_scraper(output_dir: str = "data"):
    all_items = []
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for page in tqdm(range(MAX_PAGES), desc="Pages"):
        start = page * 12
        page_items = get_page_items(start)
        if not page_items:
            logger.info(f"Stopping — no items on page {page}")
            break

        for item in tqdm(page_items, desc=f"Details page {page+1}", leave=False):
            enrich_detail_page(item)
            all_items.append(item)
            time.sleep(SCRAPE_DELAY_MIN + (SCRAPE_DELAY_MAX - SCRAPE_DELAY_MIN) * 0.3)  # slight variation

        time.sleep(3.2)

    logger.info(f"Collected {len(all_items)} assessments")

    timestamp = time.strftime("%Y%m%d_%H%M")
    filename = output_dir / f"shl_individual_tests_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved: {filename}")
    return str(filename)