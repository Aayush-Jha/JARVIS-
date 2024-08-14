import json
import pickle
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the intents data
with open("intents.json") as file:
    data = json.load(file)

# Load the trained model
model = load_model("chat_model.h5")  # Ensure the model name matches what was saved

# Load the tokenizer and label encoder
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define the max_len for padding based on what was used during training
max_len = 20

# Start the chat loop
while True:
    input_text = input("Enter your command-> ").strip()

    if input_text.lower() in ["quit", "exit"]:
        print("Exiting the chat. Goodbye!")
        break

    # Preprocess the input text
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, truncating='post')

    # Predict the intent
    result = model.predict(padded_sequences)
    tag = label_encoder.inverse_transform([np.argmax(result)])

    # Find and print a response
    for intent in data['intents']:
        if intent['tag'] == tag[0]:
            print(random.choice(intent['responses']))
            break