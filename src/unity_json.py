import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import glob
from typing import List, Dict, Any

def find_json_files(data_dir: str) -> List[str]:
    """Find all JSON files in the specified directory and its subdirectories."""
    json_pattern = os.path.join(data_dir, "**/*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    return json_files


def read_and_merge_json_files(data_dir: str) -> List[Dict[str, Any]]:
    """Read all JSON files and merge their contents into a single list."""
    all_records = []
    for file_path in find_json_files(data_dir):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                for record in records:
                    # Map JSON file fields to Milvus fields
                    record["episode_id"] = record.pop("episode") if "episode" in record else ""
                    record["URL"] = record.pop("url") if "url" in record else ""
                    all_records.append(record)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    print(f"Processed {len(all_records)} total records")
    return all_records
