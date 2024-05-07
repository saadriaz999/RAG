QDRANT_DATABASE_CONNECTION_STRING = "http://localhost:6333"
# QDRANT_DATABASE_CONNECTION_STRING = "http://db:6333"

GEMINI_API_KEY = ''


COLLECTIONS = {
    'gemini-2000-cosine': {
        'embedding_size': 768,
        'model_id': '01'
    },
    'gemini-2000-euclidean': {
        'embedding_size': 768,
        'model_id': '02'
    },
    'gemini-760-cosine': {
        'embedding_size': 768,
        'model_id': '03'
    },
    'gemini-760-euclidean': {
        'embedding_size': 768,
        'model_id': '04'
    },
    'gpt4all-2000-cosine': {
        'embedding_size': 384,
        'model_id': '05'
    },
    'gpt4all-2000-euclidean': {
        'embedding_size': 384,
        'model_id': '06'
    },
    'gpt4all-760-cosine': {
        'embedding_size': 384,
        'model_id': '07'
    },
    'gpt4all-760-euclidean': {
        'embedding_size': 384,
        'model_id': '08'
    }
}
