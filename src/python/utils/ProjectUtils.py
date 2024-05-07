import os
import json
import pathlib
# from gensim.models import KeyedVectors


def get_root_dir_path():
    return pathlib.Path(__file__).parent.parent.parent.parent.resolve()


def get_pubmed_json_data_path():
    return os.path.join(get_root_dir_path(), 'data', 'pubmed_data.json')


def get_preprocessed_pubmed_json_data_path(chunk_size):
    return os.path.join(get_root_dir_path(), 'data', f'preprocessed_pubmed_data_{chunk_size}.json')
 

def load_preprocessed_pubmed_data(chunk_size):
    path = get_preprocessed_pubmed_json_data_path(chunk_size)
    with open(path, 'r') as file:
        return json.load(file)


def get_embeddings_path(model_id):
    return os.path.join(get_root_dir_path(), 'data', 'embeddings', f'embeddings_{model_id}.json')


# def load_word2vec_embeddings():
#     root_path = get_root_dir_path()
#     embedding_path = os.path.join(root_path, 'data', 'word-2-vec-model', 'word2vec-google-news-300.gz')
#     model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
#     return model


def load_embeddings(chunk_size):
    root_path = get_root_dir_path()
    embedding_path = os.path.join(root_path, 'data', 'embeddings', f'embeddings_{chunk_size}.json')
    with open(embedding_path, 'r') as file:
        return json.load(file)
