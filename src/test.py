import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.process_asot_episode import process_asot_episode, extract_episode_number
from src.unity_json import read_and_merge_json_files
from src.MilvusClientASOT import MilvusClientASOT

milvus_client = MilvusClientASOT()