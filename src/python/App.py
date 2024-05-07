import socket
from flask import Flask, request, render_template

from utils import ChatbotUtils, EmbeddingUtils, QuestionAnswerUtils


socket.getaddrinfo('localhost', 5000)

CHATBOT_MODEL_NAME = 'gemini'
EMBEDDING_MODEL_NAME = 'gemini'

COLLECTION_NAME = 'gemini-2000-cosine'
TOP_K_RESULTS_FROM_EMBEDDING_SEARCH = 3

CHATBOT_MODEL = ChatbotUtils.load_chatbot(CHATBOT_MODEL_NAME)
EMBEDDING_MODEL = EmbeddingUtils.load_embedding_model(EMBEDDING_MODEL_NAME)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('search.html')


@app.route('/search', methods=['POST'])
def search():
    user_input = request.form['user_input']
    print(user_input)

    prompt = QuestionAnswerUtils.rag(user_input, COLLECTION_NAME, top_k=TOP_K_RESULTS_FROM_EMBEDDING_SEARCH)
    # return render_template('search.html')
    return prompt


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=True)
