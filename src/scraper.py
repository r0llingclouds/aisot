import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firecrawl.firecrawl import FirecrawlApp

import os
import time

app = FirecrawlApp(api_key=os.getenv("FIREWCRAWL_API_KEY"))

def scrape_url_with_retry(url, max_retries=3, delay=5):
    """Scrape a URL with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Scraping {url} (Attempt {attempt + 1}/{max_retries})")
            scrape_status = app.scrape_url(
                url,
                params={'formats': ['markdown']}
            )
            
            # Check if the scrape was successful
            if scrape_status and 'markdown' in scrape_status:
                return scrape_status
            
            print(f"Scrape incomplete, retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    print(f"Failed to scrape {url} after {max_retries} attempts")
    return None