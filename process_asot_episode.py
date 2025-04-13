import re
import json
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from song_parser import parse_songs_with_claude
from scraper import scrape_url_with_retry

def process_asot_episode(url: str, output_dir: str = None, max_retries: int = 3, delay: int = 5) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Process an A State of Trance episode URL by:
    1. Extracting the episode number from the URL
    2. Scraping the webpage content
    3. Saving the raw markdown content to a file
    4. Parsing the song list using Claude
    5. Saving the parsed results to a JSON file
    
    Args:
        url: The URL of the ASOT episode
        output_dir: Directory to save output files (if None, uses current directory)
        max_retries: Maximum number of retry attempts for scraping
        delay: Delay between retry attempts in seconds
        
    Returns:
        Tuple containing:
        - List of dictionaries containing the parsed song data
        - Path to the saved markdown file
        - Path to the saved JSON file
    """
    # Extract episode number from URL
    episode = extract_episode_number(url)
    if not episode:
        raise ValueError(f"Could not extract episode number from URL: {url}")
    
    print(f"Processing ASOT Episode {episode} from URL: {url}")
    print("Step 1: Episode number extracted.")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output filenames
    markdown_filepath = os.path.join(output_dir, f"asot_episode_{episode}_raw.md")
    json_filepath = os.path.join(output_dir, f"asot_episode_{episode}.json")
    
    # Scrape the web content
    print("Step 2: Starting web scraping...")
    scrape_result = scrape_url_with_retry(url, max_retries=max_retries, delay=delay)
    if not scrape_result or 'markdown' not in scrape_result:
        raise ValueError(f"Failed to scrape content from URL: {url}")
    print("Step 2: Web scraping completed.")
    
    raw_text = scrape_result['markdown']
    
    # Save raw markdown to file
    print("Step 3: Saving raw markdown content...")
    with open(markdown_filepath, 'w', encoding='utf-8') as f:
        f.write(raw_text)
    print(f"Step 3: Saved raw markdown to {markdown_filepath}")
    
    # Parse the song list
    print("Step 4: Parsing song list using Claude...")
    parsed_songs = parse_songs_with_claude(raw_text, episode, url)
    print("Step 4: Song list parsing completed.")
    
    # Save parsed data to JSON file
    print("Step 5: Saving parsed data to JSON...")
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(parsed_songs, f, indent=2, ensure_ascii=False)
    print(f"Step 5: Saved parsed song data to {json_filepath}")
    
    print(f"Successfully processed ASOT Episode {episode}.")
    return parsed_songs, markdown_filepath, json_filepath

def extract_episode_number(url: str) -> str:
    """
    Extract the episode number from an ASOT URL.
    
    Handles various URL formats:
    - https://www.astateoftrance.com/episode-313/
    - https://www.astateoftrance.com/a-state-of-trance-episode-870/
    - https://www.astateoftrance.com/asot-1110/
    - https://www.astateoftrance.com/listen-now-asot1119/
    
    Args:
        url: The URL of the ASOT episode
        
    Returns:
        The episode number as a string, or None if it couldn't be extracted
    """
    # Try various regex patterns based on the provided URL formats
    patterns = [
        r'episode-(\d+)',              # For format: episode-313
        r'episode[_-]?(\d+)',          # For format: episode_870 or episode-870
        r'asot[_-]?(\d+)',             # For format: asot-1110 or asot_1110
        r'asot(\d+)'                   # For format: asot1119
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None