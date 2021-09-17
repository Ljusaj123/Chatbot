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

intents_file = open('intents1.json').read() #otvaranje JSON datoteke
data = json.loads(intents_file)

words = [] # prazna lista riječi
classes = [] #prazna lista oznaka
documents = [] #prazna lista uzoraka s pridruženim oznakama
ignore_letters = ['!', '?', ',', '.', 'I', 'you', 'me', 'a', 'an'] #znakovi koji će se izbacivati iz liste words

# rastavljanje na liste
for intent in data['intents']:
    for pattern in intent['patterns']:
        # tokeniziranje, lematiziranje i uklanjanje interpukcijskih znakova
        word = nltk.word_tokenize(pattern)
        word = [lemmatizer.lemmatize(w.lower(), pos='v') for w in word if w not in ignore_letters]
        words.extend(word) # dodavanje riječi
        documents.append((word, intent['tag'])) #dodavanje parova uzoraka i oznaka
        classes.append(intent['tag'])  # dodavanje oznaka u listu classes

# sortiranje oznaka i uklanjanje duplikata
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

#spremanje listi u pickle datoteku
pickle.dump(words, open('words1.pkl', 'wb'))
pickle.dump(classes, open('classes1.pkl', 'wb'))


print(len(documents), "documents", documents)
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

#pretvaranje parova iz documents u numerički oblik
training = []
# vreća riječi za svaki uzorak
for doc in documents:
    bag = list([0] * len(words))
    output_row = list([0] * len(classes))
    pattern_words = doc[0]
    # stvaranje vreće riječi za trenutni obrazac
    for word in words: # izlaz je 0 za svaku oznaku, a 1 za trenutnu oznaku
        bag.append(1) if word in pattern_words else bag.append(0)

    #vreća riječi za trenutni uzorak
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# pretvaranje u NumPy listu
random.shuffle(training)
training = np.array(training)

# odvajanje podataka na podatke za treniranje i testiranje, X-> uzorak, Y-> oznaka
training_data = round(len(training)*0.7) #70% za trening

train_x = list(training[:training_data, 0])
train_y = list(training[:training_data, 1])
test_x1 = list(training[training_data:, 0])
test_y1 = list(training[training_data:, 1])

#spremanje datoteka test_x i test_y

pickle.dump(test_x1, open('test_x1.pkl', 'wb'))
pickle.dump(test_y1, open('test_y1.pkl', 'wb'))

print("Training data created")

# Stvaranje modela
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#postavljanje optimizatora
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#treniranje modela
hist = model.fit(train_x, train_y, epochs=1000, batch_size=4, verbose=1)
score = model.evaluate(test_x1, test_y1, batch_size=4, verbose=1)


model.save('chatbot_model1.h5', hist) #mode21 91 x4 i y4, classes2 i words2
print("model created")






