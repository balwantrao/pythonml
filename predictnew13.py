import random
import json
import pickle
import numpy as np
import nltk
import torch
import torch.nn as nn
from nltk.stem import WordNetLemmatizer
import chardet
from itertools import zip_longest

lemmatizer = WordNetLemmatizer()

random.seed(42)

lemmatizer = WordNetLemmatizer()

# Read the intents JSON file
with open('dat.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Creating empty lists to store data
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Separating words from patterns
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        
        # Associating patterns with respective tags
        documents.append((word_list, intent['tag']))

        # Appending the tags to the class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Storing the root words or lemma
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Saving the words and classes list to binary files
with open('words.pkl', 'wb') as file:
    pickle.dump(words, file)
    
with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

# Prepare the training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Making a copy of the output_empty
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Pad the sequences to have the same length
training_padded = [list(zip_longest(*training, fillvalue=0))]

# Convert training data to PyTorch tensors
train_x = torch.Tensor(training_padded[0][0])
train_y = torch.Tensor(training_padded[0][1])

# Initialize the model and define loss and optimizer
input_size = len(train_x[0])
hidden_size = 128
output_size = len(train_y[0])


class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out
# Load the PyTorch model
model = ChatbotModel(input_size, hidden_size, output_size)  # Replace with your model architecture
model.load_state_dict(torch.load('mymodel.pth'))
model.eval()  # Set the model to evaluation mode



def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)
                    for word in sentence_words]
    return sentence_words

def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 0
    return np.array(bag)

def predict_class(sentence):
    bow = bagw(sentence)
    bow = torch.Tensor(bow)  # Convert to a PyTorch tensor
    bow = bow.unsqueeze(0)   # Add a batch dimension
    res = model(bow)
    probabilities = torch.softmax(res, dim=1)
    max_probability, predicted_class_index = torch.max(probabilities, dim=1)
    predicted_class_index = predicted_class_index.item()
    
    ERROR_THRESHOLD = 0.25
    
    if max_probability > ERROR_THRESHOLD:
        intent = classes[predicted_class_index]
        return [{'intent': intent, 'probability': str(max_probability)}]
    else:
        return [{'intent': 'unknown', 'probability': str(max_probability)}]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Chatbot is up!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
