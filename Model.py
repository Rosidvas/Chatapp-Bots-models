import nltk
import json
import numpy as np
import pickle 
import tensorflow as tf
from tensorflow import keras
from nltk.stem.lancaster import LancasterStemmer
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Initialize the Lancaster Stemmer
stemmer = LancasterStemmer()
nltk.download('punkt')

# Load the JSON data
with open("Bot1_Dialogue_v1.json", "r") as file:
    data = json.load(file)

# Extract intents and initialize lists
intents = data["intents"]
words = []
classes = []
documents = []
ignore_words = ["?"]

# Preprocess the data
for intent in intents:
    for pattern in intent["patterns"]:
        # Tokenize words in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add pattern and tag to the documents
        documents.append((w, intent["tag"]))
        # Add tag to the classes list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Stem and lowercase the words and removing ignored words
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Prepare training data
training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Define the neural network model
input_layer = Input(shape=(len(training[0]),))
dense1 = Dense(8, activation="relu")(input_layer)
dense2 = Dense(8, activation="relu")(dense1)
output_layer = Dense(len(output[0]), activation="softmax")(dense2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train the model
model.fit(training, output, epochs=1000, batch_size=8, verbose=1)

# Save the model
model.save("Lauk_v1.h5")

# Save other data using Pickle

data_to_save = {"words": words, "classes": classes}
with open("LaukV1.pkl", "wb") as file:
    pickle.dump(data_to_save, file)
