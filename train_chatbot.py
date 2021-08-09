#unos biblioteka
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import json
import pickle
import nltk


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#nltk.download('punkt')
#nltk.download('wordnet')

intents_file = open('intents.json').read() #otvaranje JSON datoteke
data = json.loads(intents_file)

words = [] # prazna lista riječi
classes = [] #prazna lista oznaka
documents = [] #prazna lista uzoraka s pridruženim oznakama
ignore_letters = ['!', '?', ',', '.'] #znakovi koji će se izbacivati iz liste words

for intent in data['intents']:
    for pattern in intent['patterns']:
        # tokeniziraj svaki uzorak
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #dodavanje parova uzoraka i oznaka
        documents.append((word, intent['tag']))
        # dodavanje oznaka u listu classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatiziranje, uklanjanje duplikata i interpukcijskih znakova
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sortiranje oznaka
classes = sorted(list(set(classes)))

#spremanje listi u pickle datoteku
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


print(len(documents), "documents", documents)
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

#stvaranje podataka za treniranje
training = []
# vreća riječi za svaki uzorak
for doc in documents:
    bag = [] # inicijalizacija vreće riječi
    pattern_words = doc[0] # tokenizirane riječi uzorka
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words if word not in ignore_letters]

    # stvaranje vreće riječi za trenutni obrazac
    for word in words: # izlaz je 0 za svaku oznaku, a 1 za trenutnu oznaku
        bag.append(1) if word in pattern_words else bag.append(0)

    #vreća riječi za trenutni uzorak
    output_row = list([0] * len(classes))
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# stvaranje NumPy liste
random.shuffle(training)
#training = random.sample(training, len(training))
training = np.array(training)

# odvajanje podataka na podatke za treniranje i testiranje, X-> obrasci, Y-> oznake
training_data = round(len(training)*0.7)

train_x = list(training[:training_data, 0])
train_y = list(training[:training_data, 1])

test_x = list(training[training_data:, 0])
test_y = list(training[training_data:, 1])
print(len(test_y))

pickle.dump(train_x, open('train_x.pkl', 'wb'))
pickle.dump(train_y, open('train_y.pkl', 'wb'))
pickle.dump(test_x, open('test_x.pkl', 'wb'))
pickle.dump(test_y, open('test_y.pkl', 'wb'))

print("Training data created")

# Stvaranje modela
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#treniranje modela
hist = model.fit(train_x, train_y, epochs=900, batch_size=5, verbose=1)
score = model.evaluate(test_x, test_y, batch_size=5, verbose = 1)

model.save('chatbot_model.h5', hist)
print("model created")





