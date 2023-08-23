import random
import json
import pickle
import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.stem import WordNetLemmatizer
from itertools import zip_longest  # For padding lists

# Set random seed for reproducibility
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

# Define the neural network model using PyTorch
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

# Initialize the model and define loss and optimizer
input_size = len(train_x[0])
hidden_size = 128
output_size = len(train_y[0])
model = ChatbotModel(input_size, hidden_size, output_size)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
epochs = 2000
for epoch in range(epochs):
    # Forward pass
    outputs = model(train_x)
    loss = criterion(outputs, torch.argmax(train_y, dim=1))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), "mymodel.pth")

# Print statement to show successful training of the Chatbot model
print("Yay!")
