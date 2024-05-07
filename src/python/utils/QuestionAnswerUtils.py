from src.python.utils import EmbeddingUtils, ChatbotUtils, DatabaseUtils
from src.python import App


def rag(user_input, collection_name, top_k=5):
    user_input_embedding = EmbeddingUtils.create_embedding(user_input, App.EMBEDDING_MODEL, App.EMBEDDING_MODEL_NAME)

    database_client = DatabaseUtils.create_db_connection()
    search_result = database_client.search(
        collection_name=collection_name,
        query_vector=user_input_embedding
    )

    search_result = search_result[:top_k]

    context = ''
    for r in search_result:
        context += r.payload['context'] + '\n\n'

    prompt = 'Answer the question below using the context below.\n\n'
    prompt += 'QUESTION:\n' + user_input + '\n\n'
    prompt += 'CONTEXT:\n' + context

    print(prompt)
    response = ChatbotUtils.use_chatbot(prompt, App.CHATBOT_MODEL, App.CHATBOT_MODEL_NAME)
    print('--------------------------')
    print(response)
    return prompt + '----------------------------------------\n\n' + response
