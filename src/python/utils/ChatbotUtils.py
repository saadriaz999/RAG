import google.generativeai as genai

from src.python.utils import Constants


def load_gemini_chatbot():
    genai.configure(api_key=Constants.GEMINI_API_KEY)

    model = genai.GenerativeModel('gemini-pro')
    return model 


def gemini_chatbot(prompt, model):
    response = model.generate_content(prompt)

    return response.text


# the below two functions are general purpose functions that can be used to load and run any chatbot based model


def load_chatbot(model_name):
    if model_name == 'gemini':
        model = load_gemini_chatbot()
    elif model_name == 'word2vec':
        model = None

    return model


def use_chatbot(prompt, model, model_name):
    if model_name == 'gemini':
        response = 'RESPONSE' + gemini_chatbot(prompt, model)
    elif model_name == 'word2vec':
        response = ''

    return response
