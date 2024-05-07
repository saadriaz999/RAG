import numpy as np
import google.generativeai as genai
# from nltk.tokenize import word_tokenize

from src.python.utils import ProjectUtils, Constants


def load_gemini_embedding():
    genai.configure(api_key=Constants.GEMINI_API_KEY)

    model = genai.GenerativeModel('gemini-pro')
    return model


def gemini_embedding(user_input):
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=user_input,
        task_type="RETRIEVAL_QUERY"
    )

    return embedding['embedding']


def word2vec_embedding(user_input, model):
    words = word_tokenize(user_input.lower())
    words = [word for word in words if word.isalpha()]  # Remove punctuation and numbers

    # Convert words to vectors and average them
    word_vectors = [model[word] for word in words if word in model]
    if len(word_vectors) > 0:
        paragraph_vector = np.mean(word_vectors, axis=0)
    else:
        paragraph_vector = np.zeros(model.vector_size)

    return list(paragraph_vector)


# the below two functions are general purpose functions that can be used to load and run any embedding model
def load_embedding_model(model_name):
    if model_name == 'gemini':
        model = load_gemini_embedding()
    elif model_name == 'word2vec':
        model = ProjectUtils.load_word2vec_embeddings()

    return model


def create_embedding(input, model, model_name):
    if model_name == 'gemini':
        embedding = gemini_embedding(input)
    elif model_name == 'word2vec':
        embedding = word2vec_embedding(input, model)

    return embedding
