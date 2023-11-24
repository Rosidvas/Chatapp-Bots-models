import nltk
import json
import pickle
import numpy as np
import random
import tensorflow as tf
import keras
from nltk.stem.lancaster import LancasterStemmer
from keras.models import load_model

stemmer = LancasterStemmer()

#Dialogue/Responses Data
with open("Bot1_Dialogue_v1.json", "r") as file:
    data = json.load(file)

intents = data["intents"]

#Chatbot model
data = pickle.load(open("LaukV1.pkl", "rb"))
words = data["words"]
classes = data["classes"]

model = load_model("Lauk_v1.h5")
model.summary()

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

