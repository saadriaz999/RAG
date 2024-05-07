import json
# import nltk
from qdrant_client.models import Distance
from langchain_text_splitters import CharacterTextSplitter

from utils import ProjectUtils, DatabaseUtils, EmbeddingUtils


def load_pubmed_dataset():
    """Load the raw pubmed dataset"""

    path = ProjectUtils.get_pubmed_json_data_path()

    with open(path, 'r') as file:
        data = json.load(file)

    return data


def create_rag_dataset(pubmed_data, chunk_size):
    """Use chunks to create a dataset along with IDs for a RAG system"""

    data = []
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator='.')
    for record in pubmed_data:
        chunks = text_splitter.create_documents(['. '.join(record['context']['contexts'])])
        for i, context in enumerate(chunks):
            data.append({
                'id': f"{record['pubid']}-{str(i).zfill(3)}",
                'context': context.page_content
            })

    return data


def save_processed_dataset(chunk_size=2000):
    """Save the preprocessed data"""

    data = load_pubmed_dataset()
    preprocessed_data = create_rag_dataset(data, chunk_size)

    path_to_save = ProjectUtils.get_preprocessed_pubmed_json_data_path(chunk_size)

    with open(path_to_save, "w") as json_file:
        json.dump(preprocessed_data, json_file)


if __name__ == '__main__':
    # # install punkt using nltk library
    # nltk.download('punkt')

    # ----------------------------------------------------------------------------------------------------

    # loading embedding models
    GEMINI_EMBEDDING_MODEL_NAME = 'gemini'
    GEMINI_EMBEDDING_MODEL = EmbeddingUtils.load_embedding_model(GEMINI_EMBEDDING_MODEL_NAME)

    # WORD2VEC_EMBEDDING_MODEL_NAME = 'word2vec'
    # WORD2VEC_EMBEDDING_MODEL = EmbeddingUtils.load_embedding_model(WORD2VEC_EMBEDDING_MODEL_NAME)

    # ----------------------------------------------------------------------------------------------------

    # save the preprocessed data to the 'data' folder with chunks of size 760
    save_processed_dataset(chunk_size=760)
    print('Saved dataset with chunk size 760.')

    # save vectors of preprocessed dataset created by gemini api, chunk_size 760
    DatabaseUtils.create_and_store_embeddings(
        GEMINI_EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL_NAME, Distance.COSINE, 'gemini-760-cosine', 760)
    print('Saved gemini vectors in database (chunk_size = 760, distance_metric = cosine similarity)')

    # save vectors of preprocessed dataset created by gemini api, chunk_size 760
    DatabaseUtils.create_and_store_embeddings(
        GEMINI_EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL_NAME, Distance.EUCLID, 'gemini-760-euclidean', 760)
    print('Saved gemini vectors in database (chunk_size = 760, distance_metric = cosine similarity)')

    # # save vectors of preprocessed dataset created by gpt4all, chunk_size 760
    # DatabaseUtils.store_embeddings(Distance.COSINE, 'gpt4all-760-cosine', 760)
    # print('Saved gpt4all vectors in database (chunk_size = 760, distance_metric = cosine similarity)')
    #
    # # save vectors of preprocessed dataset created by gpt4all, chunk_size 760
    # DatabaseUtils.store_embeddings(Distance.EUCLID, 'gpt4all-760-euclidean', 760)
    # print('Saved gpt4all vectors in database (chunk_size = 760, distance_metric = cosine similarity)')

    # ----------------------------------------------------------------------------------------------------

    # save the preprocessed data to the 'data' folder with chunks of size 2000
    save_processed_dataset(chunk_size=2000)
    print('Saved dataset with chunk size 2000.')

    # save vectors of preprocessed dataset created by gemini api, chunk_size 2000
    DatabaseUtils.create_and_store_embeddings(
        GEMINI_EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL_NAME, Distance.COSINE, 'gemini-2000-cosine', 2000)
    print('Saved gemini vectors in database (chunk_size = 2000, distance_metric = cosine similarity)')

    # save vectors of preprocessed dataset created by gemini api, chunk_size 2000
    DatabaseUtils.create_and_store_embeddings(
        GEMINI_EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL_NAME, Distance.EUCLID, 'gemini-2000-euclidean', 2000)
    print('Saved gemini vectors in database (chunk_size = 2000, distance_metric = cosine similarity)')

    # # save vectors of preprocessed dataset created by gpt4all, chunk_size 2000
    # DatabaseUtils.store_embeddings(Distance.COSINE, 'gpt4all-2000-cosine', 2000)
    # print('Saved gpt4all vectors in database (chunk_size = 2000, distance_metric = cosine similarity)')
    #
    # # save vectors of preprocessed dataset created by gpt4all, chunk_size 2000
    # DatabaseUtils.store_embeddings(Distance.EUCLID, 'gpt4all-2000-euclidean', 2000)
    # print('Saved gpt4all vectors in database (chunk_size = 2000, distance_metric = cosine similarity)')
