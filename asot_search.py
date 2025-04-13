import os
import sys
import gradio as gr
import pandas as pd
import dotenv
from pathlib import Path

# Add the root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
dotenv.load_dotenv()

# Import the MilvusClientASOT class
from src.MilvusClientASOT import MilvusClientASOT

# Initialize MilvusClient
milvus_client = MilvusClientASOT()

# Get collection name from environment or use default
collection_name = os.getenv("MILVUS_COLLECTION", "asot_songs")

def format_result(hit):
    """Format a search result into a readable dictionary"""
    result = {}
    
    # Basic information
    result["Episode"] = hit['entity']['episode_id']
    
    # Artist information
    if hit['entity']['artist'] != 'nav':
        result["Artist"] = hit['entity']['artist']
    
    if hit['entity']['collaborators'] != 'nav':
        result["Collaborators"] = hit['entity']['collaborators']
    
    if hit['entity']['featured_artists'] != 'nav':
        result["Featured Artists"] = hit['entity']['featured_artists']
    
    # Song information
    if hit['entity']['title'] != 'nav':
        result["Title"] = hit['entity']['title']
    
    if hit['entity']['remix_info'] != 'nav':
        result["Remix"] = hit['entity']['remix_info']
    
    # Ranking information
    if hit['entity']['ranking'] != -1:
        result["Ranking"] = hit['entity']['ranking']
    
    if hit['entity']['popularity_score'] != -1:
        result["Popularity Score"] = hit['entity']['popularity_score']
    
    if hit['entity']['vote_count'] != -1:
        result["Vote Count"] = hit['entity']['vote_count']
    
    # URL and score
    if hit['entity']['URL'] != 'nav':
        result["URL"] = hit['entity']['URL']
    
    result["Match Score"] = round(hit['distance'], 4)
    
    return result

def search(query, search_type, limit, sparse_weight=0.3, dense_weight=0.7, rrf_k=60):
    """Perform search on Milvus based on specified parameters"""
    if not query:
        return "Please enter a search query"
    
    try:
        # Perform search based on selected type
        if search_type == "Sparse Search (BM25)":
            results = milvus_client.sparse_search(
                collection_name=collection_name,
                query_text=query,
                limit=limit
            )
        elif search_type == "Dense Search (Vector)":
            results = milvus_client.dense_search(
                collection_name=collection_name,
                query_text=query,
                limit=limit
            )
        elif search_type == "Hybrid Search (Weighted)":
            results = milvus_client.hybrid_search(
                collection_name=collection_name,
                query_text=query,
                limit=limit,
                ranker_type="weighted",
                sparse_weight=sparse_weight,
                dense_weight=dense_weight
            )
        else:  # "Hybrid Search (RRF)"
            results = milvus_client.hybrid_search(
                collection_name=collection_name,
                query_text=query,
                limit=limit,
                ranker_type="rrf",
                k=rrf_k
            )
            
        # Format results
        formatted_results = [format_result(hit) for hit in results]
        
        # Convert to DataFrame for display
        if formatted_results:
            df = pd.DataFrame(formatted_results)
            return df
        else:
            return "No results found"
    
    except Exception as e:
        return f"Error performing search: {str(e)}"

def get_collection_stats():
    """Get statistics about the collection"""
    try:
        if not milvus_client.client.has_collection(collection_name):
            return f"Collection '{collection_name}' does not exist"
        
        stats = milvus_client.get_collection_stats(collection_name)
        episodes = milvus_client.list_episodes(collection_name)
        
        stats_text = f"Collection: {collection_name}\n"
        stats_text += f"Total Songs: {stats.get('row_count', 0)}\n"
        stats_text += f"Episodes: {len(episodes)}\n"
        stats_text += f"Episode Numbers: {', '.join(episodes)}"
        
        return stats_text
    except Exception as e:
        return f"Error getting collection stats: {str(e)}"

# Create Gradio Interface
with gr.Blocks(title="ASOT Song Search") as demo:
    gr.Markdown("# A State of Trance - Song Search")
    gr.Markdown("Search for songs across ASOT episodes using vector search")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Search inputs
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter artist, song title, or any keywords",
                lines=1
            )
            
            search_type = gr.Radio(
                label="Search Method",
                choices=[
                    "Sparse Search (BM25)",
                    "Dense Search (Vector)",
                    "Hybrid Search (Weighted)",
                    "Hybrid Search (RRF)"
                ],
                value="Hybrid Search (Weighted)"
            )
            
            limit_slider = gr.Slider(
                label="Number of Results",
                minimum=1,
                maximum=50,
                value=10,
                step=1
            )
            
            with gr.Accordion("Advanced Search Parameters", open=False):
                with gr.Row():
                    sparse_weight = gr.Slider(
                        label="Sparse Weight",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05
                    )
                    
                    dense_weight = gr.Slider(
                        label="Dense Weight",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.05
                    )
                
                rrf_k = gr.Slider(
                    label="RRF K Value",
                    minimum=1,
                    maximum=100,
                    value=60,
                    step=1
                )
            
            search_button = gr.Button("Search")
        
        with gr.Column(scale=1):
            # Collection info
            stats_output = gr.Textbox(
                label="Collection Stats",
                interactive=False,
                lines=6
            )
            
            refresh_stats = gr.Button("Refresh Stats")
    
    # Results display
    results_output = gr.DataFrame(label="Search Results")
    
    # Event handlers
    search_button.click(
        fn=search,
        inputs=[
            query_input,
            search_type,
            limit_slider,
            sparse_weight,
            dense_weight,
            rrf_k
        ],
        outputs=results_output
    )
    
    refresh_stats.click(
        fn=get_collection_stats,
        inputs=[],
        outputs=stats_output
    )
    
    # Load stats on first load
    demo.load(
        fn=get_collection_stats,
        inputs=[],
        outputs=stats_output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)