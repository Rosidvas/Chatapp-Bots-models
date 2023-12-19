from flask import Flask, jsonify, request
import nltk
import json
import pickle
import numpy as np
import random
from nltk.stem.lancaster import LancasterStemmer
from keras.models import load_model

app = Flask(__name__)

stemmer = LancasterStemmer()

# Load necessary data and models
with open("Bot1_Dialogue_v1.json", "r") as file:
    data = json.load(file)

intents = data["intents"]

data = pickle.load(open("LaukV1.pkl", "rb"))
words = data["words"]
classes = data["classes"]

model = load_model("Lauk_v1.h5")

def preprocess_input(user_input):
    tokenized_input = nltk.word_tokenize(user_input)  
    stemmed_input = [stemmer.stem(word.lower()) for word in tokenized_input]
    return stemmed_input

def get_response(intent_index):
    recognized_intent = classes[intent_index]
    for intent in intents:
        if intent["tag"] == recognized_intent:
            response = random.choice(intent["responses"])
            break
    return response

@app.route('/analyze_user_input', methods=['POST'])
def handle_analyze_user_input():
    data = request.get_json()
    username = data.get('username')
    user_input = data.get('user_input')
    
    if not (username and user_input):
        return jsonify({'response': 'Invalid input data.'})
    
    response = analyze_user_input(username, user_input)
    return jsonify({'response': response})

def analyze_user_input(username, user_input):      
    preprocessed_input = preprocess_input(user_input)
    
    if preprocessed_input == False:
        return False
            
    input_data = [0] * len(words)
    for word in preprocessed_input:
        if word in words:
            input_data[words.index(word)] = 1
    input_data = np.array(input_data).reshape(1, -1)
    predicted_output = model.predict(input_data)[0]
    intent_index = np.argmax(predicted_output)
    response = get_response(intent_index).format(user=username)
    return response

if __name__ == '__main__':
    app.run(debug=True)
