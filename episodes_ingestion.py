from process_asot_episode import process_asot_episode, extract_episode_number
from unity_json import read_and_merge_json_files
from MilvusClientASOT import MilvusClientASOT
import os

milvus_client = MilvusClientASOT()

episodes = []

with open('episodes_to_insert.txt', 'r') as file:
    episodes = file.readlines()
    # remove possible \n
    episodes = [episode.strip() for episode in episodes]

collection_name = os.getenv("MILVUS_COLLECTION")
data_dir = os.getenv("OUTPUT_FOLDER")

if milvus_client.client.has_collection(collection_name):
    existing_episodes = milvus_client.list_episodes(collection_name)
else:
    existing_episodes = []

episodes_to_process =  [episode for episode in episodes if extract_episode_number(episode) not in existing_episodes]

results = []
for url in episodes_to_process:
    try:
        result = process_asot_episode(url, output_dir=data_dir)
        results.append(result)
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

all_records = read_and_merge_json_files(data_dir) # ready to be loaded into milvus

milvus_client.create_collection_if_not_exists(collection_name)

milvus_client.insert_episodes(collection_name, all_records)