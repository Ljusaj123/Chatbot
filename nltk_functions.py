import nltk
from nltk.stem import WordNetLemmatizer
import webbrowser
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
import json
import random

from keras.models import load_model
model = load_model('chatbot_model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    ignore_letters = ['!', '?', ',', '.']
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_letters]
    return sentence_words


# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return (np.array(bag))


def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.85
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    if not results:
        return []

    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print(return_list)
    return return_list


def getResponse(ints, intents_json):

    if ints==[]:
        result = random.choice(intents['intents'][4]['responses'])
        print("ERROR:RETURNED THRESHOLD")
        return result

    else:
        list_of_intents = intents_json['intents']
        tag = ints[0]['intent']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                if i['url']:
                    print(i['url'])
                    print(str(i['url']))
                    webbrowser.open_new(i['url'])
                break
        return result