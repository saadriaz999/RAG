import json
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams
from qdrant_client.models import PointStruct

from src.python.utils import Constants, ProjectUtils, EmbeddingUtils


def create_db_connection():
    """Establishes a connection to the QDRant Vector database and return a
    client to communicate with the database"""
    
    client = QdrantClient(url=Constants.QDRANT_DATABASE_CONNECTION_STRING)
    return client


def create_and_store_embeddings(model, model_name, distance_metric, collection_name, chunk_size):
    """Creates embedding of dataset for given model and stores in database"""

    data = ProjectUtils.load_preprocessed_pubmed_data(chunk_size)

    embedding_size = Constants.COLLECTIONS[collection_name]['embedding_size']
    model_id = Constants.COLLECTIONS[collection_name]['model_id']
    path = ProjectUtils.get_embeddings_path(model_id)

    data = data[:100]  # first 98 documents, # 3627, 5422,

    ids = [f"{entry['id']}-{model_id}" for entry in data]
    vectors = [EmbeddingUtils.create_embedding(entry['context'], model, model_name) for entry in data]
    payloads = [{'context': entry['context']} for entry in data]

    points = [PointStruct(id=i, payload=payloads[i], vector=vectors[i]) for i in range(len(ids))]
    vector_dictionary = {key: value for key, value in zip(ids, vectors)}

    database_client = create_db_connection()
    try:
        database_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=distance_metric),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
            shard_number=4
        )
    except:
        pass

    database_client.upload_points(
        collection_name=collection_name,
        wait=True,
        points=points
    )

    database_client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )

    with open(path, "w") as json_file:
        json.dump(vector_dictionary, json_file)


def store_embeddings(distance_metric, collection_name, chunk_size):
    """Stores embedding of dataset for given model in database"""

    embedding_size = Constants.COLLECTIONS[collection_name]['embedding_size']
    model_id = Constants.COLLECTIONS[collection_name]['model_id']
    data = ProjectUtils.load_embeddings(chunk_size)

    # data = data[:500]  # first 98 documents, # 3627, 5422,

    ids = [f"{entry['id']}-{model_id}" for entry in data]
    vectors = [entry['embedding'] for entry in data]
    payloads = [{'context': entry['context']} for entry in data]

    points = [PointStruct(id=i, payload=payloads[i], vector=vectors[i]) for i in range(len(ids))]

    database_client = create_db_connection()
    try:
        database_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=distance_metric),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
            shard_number=4
        )
    except:
        pass

    database_client.upload_points(
        collection_name=collection_name,
        wait=True,
        points=points
    )

    database_client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )
