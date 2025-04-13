from pymilvus import MilvusClient as MC
from pymilvus import AnnSearchRequest
from pymilvus import WeightedRanker, MilvusException, RRFRanker

from pymilvus import Function, FunctionType
from pymilvus import DataType
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

from Logger import Logger
from Singleton import Singleton
import json

import os

class MilvusClientASOT(metaclass=Singleton):
    """
    MilvusClient is a singleton class that provides a client for the Milvus database.
    It provides methods to create collections, insert data, and perform searches.
    """

    def __init__(self):

        self.logger = Logger('milvus_logger', os.getenv("LOG_MISC", "DEBUG")).logger

        self.client = MC(host="localhost", port="19530")

        self.logger.debug("Milvus Client successfully initialized.")

        self.embeddings = SentenceTransformerEmbeddingFunction("intfloat/e5-large-v2")
        self.sparse_embedding_function = None
        self.logger.debug("Dense embeddings initialized.")
        
    def create_schema(self, auto_id=True, enable_dynamic_field=True):
        """
        Create a schema for a Milvus collection with required fields and BM25 function.
        
        Args:
            auto_id (bool): Whether to auto-generate IDs
            enable_dynamic_field (bool): Whether to enable dynamic fields
        
        Returns:
            CollectionSchema: A complete schema object for the collection
        """
        
        # Create schema
        schema = self.client.create_schema(
            auto_id=auto_id,
            enable_dynamic_field=enable_dynamic_field,
        )


        # # Add fields to schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        
        # Job position specific fields
        schema.add_field(field_name="episode_id", datatype=DataType.VARCHAR, max_length=50)
                
        # Additional song metadata fields
        schema.add_field(field_name="ranking", datatype=DataType.INT64)
        schema.add_field(field_name="artist", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="collaborators", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="featured_artists", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="remix_info", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="popularity_score", datatype=DataType.INT64)
        schema.add_field(field_name="vote_count", datatype=DataType.INT64)

        # Test will be a composite field made of the other fields
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=10000, enable_analyzer=True)
        
        # Add the new URL field
        schema.add_field(field_name="URL", datatype=DataType.VARCHAR, max_length=2048) # Assuming a max length for URL

        # Vector fields for search
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=self.embeddings.dim) 

        # Define function to generate sparse vectors
        bm25_function = Function(
            name="text_bm25_emb", # Function name
            input_field_names=["text"], # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,
        )
        
        schema.add_function(bm25_function)
        
        return schema
    
    def create_indices(self, collection_name):
        """
        Prepare index parameters for both dense and sparse vector fields.
        
        Args:
            collection_name (str): Name of the collection for index preparation
        """
        # Prepare index parameters
        index_params = self.client.prepare_index_params()
        
        # Add indexes
        index_params.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        
        index_params.add_index(
            field_name="sparse",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
            metric_type="BM25",  # Set to `BM25` when using function to generate sparse vectors
            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm for sparse index
        )
        
        return index_params
    
    def create_collection(self, collection_name, schema, index_params):
        """
        Create a new collection in Milvus with the provided schema and index parameters.
        
        Args:
            collection_name (str): Name of the collection
            schema (CollectionSchema): Schema for the collection
            index_params: Index parameters for the collection
        """
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        
        self.logger.debug(f"Created collection: {collection_name}")

    def prepare_data_for_insertion(self, documents):
        """
        Prepare documents for insertion into Milvus by generating dense embeddings.
        
        Args:
            documents (list): List of dictionaries with job position fields
            
        Returns:
            list: List of documents ready for insertion with dense embeddings
        """
        prepared_data = []
        
        # Define fields to include in the concatenated text field
        text_fields = ['episode_id', 'ranking', 'artist', 'collaborators', 'featured_artists', 'title', 'remix_info', 'popularity_score', 'vote_count', 'URL']

        # Extract text and prepare data points
        docs_texts_to_embed = []
        for doc in documents:
            # Construct the text field by concatenating specified fields
            text_parts = []
            for field in text_fields:
                value = doc.get(field)
                # Only include non-null, non-default values
                if value is not None and value != 'nav' and value != -1:
                    text_parts.append(str(value))
            
            constructed_text = " ".join(text_parts)
            docs_texts_to_embed.append("query: " + constructed_text)

            # Prepare the data point without the 'text' field initially
            # Explicitly handle None values for each field
            data_point = {
                "episode_id": doc.get('episode_id') if doc.get('episode_id') is not None else 'nav',
                "ranking": doc.get('ranking') if doc.get('ranking') is not None else -1,
                "artist": doc.get('artist') if doc.get('artist') is not None else 'nav',
                "collaborators": doc.get('collaborators') if doc.get('collaborators') is not None else 'nav',
                "featured_artists": doc.get('featured_artists') if doc.get('featured_artists') is not None else 'nav',
                "title": doc.get('title') if doc.get('title') is not None else 'nav',
                "remix_info": doc.get('remix_info') if doc.get('remix_info') is not None else 'nav',
                "popularity_score": doc.get('popularity_score') if doc.get('popularity_score') is not None else -1,
                "vote_count": doc.get('vote_count') if doc.get('vote_count') is not None else -1,
                "URL": doc.get('URL') if doc.get('URL') is not None else 'nav'
            }
            prepared_data.append(data_point)
         # dump prepared_data to a json file for debugging
         
        with open('prepared_data.json', 'w') as f:
            json.dump(prepared_data, f)

        # Generate dense embeddings in batch using the constructed texts
        dense_vectors = self.embeddings(docs_texts_to_embed)
        
        # Add dense vectors and the constructed text back to the prepared data
        for i, data_point in enumerate(prepared_data):
            data_point["dense"] = dense_vectors[i]
            # Extract the original constructed text (without "query: ")
            data_point["text"] = docs_texts_to_embed[i][len("query: "):] 
        
        return prepared_data

    def insert_data(self, collection_name, prepared_data):
        """
        Insert data into a collection. Always prepares embeddings before insertion.
        
        Args:
            collection_name (str): Name of the collection
            data (list): List of dictionaries with 'id' and 'text' fields
            
        Returns:
            The result of the insert operation
        """

        self.logger.debug(f"Prepared {len(prepared_data)} documents with embeddings")
        
        # Insert prepared data
        res = self.client.insert(
            collection_name=collection_name,
            data=prepared_data
        )
        self.logger.debug(f"Inserted {len(prepared_data)} documents into {collection_name}")
        
        return res

    def create_and_load_collection(self, collection_name, documents):
        """
        Create a collection and load it with documents in one operation.
        
        Args:
            collection_name (str): Name of the collection
            documents (list): List of dictionaries with 'id' and 'text' fields
            
        Returns:
            dict: Result of the operation
        """
        # Check if collection exists
        if self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} already exists. Delete it before recreating it.")
        
        # Create schema and indices
        schema = self.create_schema()
        index_params = self.create_indices(collection_name)
        
        # Create collection
        self.create_collection(collection_name, schema, index_params)
        
        # Insert data
        prepared_data = self.prepare_data_for_insertion(documents)
        result = self.insert_data(collection_name, prepared_data)
        
        return result

    def list_collections(self) -> list:
        """
        List all collections in the Milvus instance.
        
        Returns:
            list: List of collection names
        
        Raises:
            MilvusException: If there was an error retrieving the collections
        """
        try:
            collections = self.client.list_collections()
            self.logger.debug(f"Found {len(collections)} collections")
            return collections
            
        except MilvusException as e:
            self.logger.error(f"Failed to list collections: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while listing collections: {str(e)}")
            raise

    def get_collection_stats(self, collection_name: str) -> dict:
        """
        Get detailed statistics about a collection including row count, fields and their types.
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            dict: Complete collection statistics and metadata
        """
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist")
            
        try:
            # Get row count from stats
            stats = self.client.get_collection_stats(collection_name)
            
            # Get full collection description
            description = self.client.describe_collection(collection_name)
            
            # Combine the information
            result = description.copy()
            result["row_count"] = stats.get("row_count", 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def list_episodes(self, collection_name: str) -> list[str]:
        """
        Retrieve a list of unique episode IDs present in the collection.

        Args:
            collection_name (str): The name of the collection to query.

        Returns:
            list[str]: A sorted list of unique episode IDs found in the collection.

        Raises:
            ValueError: If the collection does not exist.
            MilvusException: If there is an error during the query.
        """
        if not self.client.has_collection(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist.")
            raise ValueError(f"Collection {collection_name} does not exist")

        try:
            # Query all entities, fetching only the episode_id field.
            # Assuming 'id' is the primary key and auto_id=True as defined in create_schema
            results = self.client.query(
                collection_name=collection_name,
                filter="id >= 0",
                output_fields=["episode_id"],
            )

            # Extract unique episode IDs using a set comprehension and sort them
            episode_ids = {item['episode_id'] for item in results if 'episode_id' in item}
            unique_episodes = sorted(list(episode_ids))

            self.logger.debug(f"Found {len(unique_episodes)} unique episodes in collection {collection_name}")
            return unique_episodes

        except MilvusException as e:
            self.logger.error(f"Failed to query episode IDs from {collection_name}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while querying episode IDs: {str(e)}")
            raise

    def delete_collection(self, collection_name):
        """
        Delete a collection from Milvus.
        """
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            self.logger.info(f"Collection {collection_name} deleted")
        else:
            self.logger.warning(f"Collection {collection_name} does not exist")
        
    def dense_search(self, collection_name, query_text, limit=5):
        """
        Perform a dense vector search using the query text.
        
        Args:
            collection_name (str): Name of the collection to search
            query_text (str): Text query to generate dense embedding
            limit (int, optional): Maximum number of results. Defaults to 10.
        
        Returns:
            list: List of search results with job position data
        """
        query_vector = self.embeddings(["query: " + query_text])[0]
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="dense",
            limit=limit,
            output_fields=["episode_id", "text", "ranking", "artist", "collaborators", "featured_artists", "title", "remix_info", "popularity_score", "vote_count", "URL"],
            search_params=search_params
        )[0]
        return results
    
    def sparse_search(self, collection_name, query_text, limit=5):
        """
        Perform a sparse vector search using the query text with BM25.
        
        Args:
            collection_name (str): Name of the collection to search
            query_text (str): Text query for sparse search
            limit (int, optional): Maximum number of results. Defaults to 10.
        
        Returns:
            list: List of search results with job position data
        """
        search_params = {"metric_type": "BM25", "params": {}}
        results = self.client.search(
            collection_name=collection_name,
            data=[query_text],
            anns_field="sparse",
            limit=limit,
            output_fields=["episode_id", "text", "ranking", "artist", "collaborators", "featured_artists", "title", "remix_info", "popularity_score", "vote_count", "URL"],
            search_params=search_params
        )[0]
        return results

    def hybrid_search(self, collection_name, query_text, limit=5, ranker_type="weighted", **kwargs):
        """
        Perform a hybrid search combining dense and sparse vector searches.
        More info: https://milvus.io/docs/multi-vector-search.md
        
        Args:
            collection_name (str): Name of the collection to search
            query_text (str): Text query for generating dense embedding and sparse search
            ranker_type (str): Type of ranker to use ('weighted' or 'rrf')
            limit (int, optional): Maximum number of results. Defaults to 10.
            **kwargs: Parameters for the specific ranker:
                - If ranker_type is 'weighted': sparse_weight (default=0.3), dense_weight (default=0.7)
                - If ranker_type is 'rrf': k (default=60)
        
        Returns:
            list: List of search results with job position data
        """
        sparse_search_param = {
            "data": [query_text],
            "anns_field": "sparse",
            "param": {"metric_type": "BM25", "params": {}},
            "limit": limit
        }
        sparse_req = AnnSearchRequest(**sparse_search_param)

        query_dense_vector = self.embeddings(["query: " + query_text])[0]
        dense_search_param = {
            "data": [query_dense_vector],
            "anns_field": "dense",
            "param": {"metric_type": "IP", "params": {}},
            "limit": limit
        }
        dense_req = AnnSearchRequest(**dense_search_param)

        # Create appropriate ranker based on type
        if ranker_type.lower() == "weighted":
            sparse_weight = kwargs.get("sparse_weight", 0.3)
            dense_weight = kwargs.get("dense_weight", 0.7)
            ranker = WeightedRanker(sparse_weight, dense_weight)
            self.logger.debug(f"Using WeightedRanker with weights {sparse_weight} and {dense_weight}")
        elif ranker_type.lower() == "rrf":
            k = kwargs.get("k", 60)
            ranker = RRFRanker(k)
            self.logger.debug(f"Using RRFRanker with k={k}")
        else:
            raise ValueError(f"Unknown ranker type: {ranker_type}")

        results = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[sparse_req, dense_req],
            ranker=ranker,
            limit=limit,
            output_fields=["episode_id", "text", "ranking", "artist", "collaborators", "featured_artists", "title", "remix_info", "popularity_score", "vote_count", "URL"]
        )[0]
        return results

    def insert_new_episodes(self, collection_name: str, documents: list) -> dict | None:
        """
        Inserts documents into the specified collection only if their episode_id 
        is not already present in the collection.

        Args:
            collection_name (str): The name of the collection.
            documents (list): A list of dictionaries representing the documents to insert. 
                              Each dictionary must contain an 'episode_id' key.

        Returns:
            dict | None: The result of the insert operation if any documents were inserted, 
                         otherwise None. Returns None if no new episodes were found to insert.

        Raises:
            ValueError: If the collection does not exist or if a document lacks 'episode_id'.
            MilvusException: If there is an error during listing episodes or insertion.
        """
        if not self.client.has_collection(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist.")
            raise ValueError(f"Collection {collection_name} does not exist")

        try:
            existing_episode_ids = set(self.list_episodes(collection_name))
            self.logger.debug(f"Found {len(existing_episode_ids)} existing episode IDs in {collection_name}.")
        except Exception as e:
            self.logger.error(f"Failed to retrieve existing episode IDs: {e}")
            raise # Re-raise after logging

        documents_to_insert = []
        skipped_count = 0
        for doc in documents:
            episode_id = doc.get('episode_id')
            if episode_id is None:
                self.logger.warning(f"Document missing 'episode_id'. Skipping: {doc}")
                skipped_count += 1
                continue

            if episode_id not in existing_episode_ids:
                documents_to_insert.append(doc)
            else:
                skipped_count += 1

        if skipped_count > 0:
             self.logger.info(f"Skipped {skipped_count} documents because their episode_id already exists or was missing.")

        if not documents_to_insert:
            self.logger.info("No new episodes found to insert.")
            return None

        try:
            prepared_data = self.prepare_data_for_insertion(documents_to_insert)
            insert_result = self.insert_data(collection_name, prepared_data)
            self.logger.info(f"Successfully inserted {len(documents_to_insert)} new episode documents into {collection_name}.")
            return insert_result
        except Exception as e:
            self.logger.error(f"Failed during data preparation or insertion for new episodes: {e}")
            raise # Re-raise after logging
