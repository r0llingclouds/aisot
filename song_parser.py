import requests
import json
import os
import re
from typing import List, Dict, Any, Optional

def parse_songs_with_claude(raw_text: str, episode: str, url: str) -> List[Dict[str, Any]]:
    """
    Parse ASOT song list using Claude LLM with robust handling of varied formats.
    
    Args:
        raw_text: Raw text with song listings in any format
        episode: The ASOT episode number or identifier.
        url: The URL source of the song list.
        
    Returns:
        List of dictionaries with parsed song data, including episode and url fields.
    """
    # Get API key from env if not provided
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Claude API key required. Set ANTHROPIC_API_KEY or pass api_key parameter.")
    
    # Create prompt for Claude - more flexible about format and fields
    prompt = f"""
    <task>
    Parse this list of songs from A State of Trance (ASOT) into structured data.
    
    The songs may be in any format and may be missing some information. Do your best to extract
    whatever data is available for each song. Common elements might include:
    
    - Ranking number (might be at the beginning)
    - Artist names (may include multiple artists, collaborations, features)
    - Song title
    - Remix information (often in parentheses)
    - Popularity/voting numbers (often at the end)
    
    For any song, extract whatever fields you can identify, which might include:
    - ranking: Any position/ranking number (if present)
    - artist: Primary artist name(s)
    - collaborators: Any artists in collaboration (if identifiable) 
    - featured_artists: Any featured artists (if identifiable)
    - title: Song title
    - remix_info: Any remix details
    - popularity_score: First number if there are numbers at the end
    - vote_count: Second number if there are numbers at the end
    
    Do not force fields that aren't clearly present. Only include fields you're confident about.
    Return a JSON array of objects with whatever fields you can reliably extract.
    </task>
    
    <song_list>
    {raw_text}
    </song_list>
    """
    
    # Call Claude API
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 4000,
        "temperature": 0,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data
    )
    response.raise_for_status()
    
    # Extract JSON from response
    result = response.json()
    content = result["content"][0]["text"]
    
    # Find JSON array in response - more flexible pattern matching
    json_pattern = r'(\[\s*\{.*?\}\s*\])'
    json_match = re.search(json_pattern, content, re.DOTALL)
    
    if not json_match:
        # Try alternative approach if the first one fails
        # Look for any text that might be JSON
        try:
            # Find the first opening bracket and the last closing bracket
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx:end_idx+1]
                songs = json.loads(json_str)
                # Add episode and url to each song entry
                for song in songs:
                    song['episode'] = episode
                    song['url'] = url
                return songs
        except:
            pass
            
        raise ValueError("Could not extract structured data from Claude response")
        
    json_str = json_match.group(0)
    
    try:
        songs = json.loads(json_str)
        # Add episode and url to each song entry
        for song in songs:
            song['episode'] = episode
            song['url'] = url
        return songs
    except json.JSONDecodeError:
        # Try to clean up the JSON if it's malformed
        cleaned_json = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        cleaned_json = re.sub(r',\s*]', ']', cleaned_json)  # Remove trailing commas in arrays
        
        try:
            songs = json.loads(cleaned_json)
            # Add episode and url to each song entry
            for song in songs:
                song['episode'] = episode
                song['url'] = url
            return songs
        except:
            raise ValueError("Failed to parse Claude's response as valid JSON")


