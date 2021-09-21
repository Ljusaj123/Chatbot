import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
import json
import random
import webbrowser

#učitavanje modela i datoteka words, classes i intents2
from keras.models import load_model
model = load_model('chatbot_model2.h5')

intents = json.loads(open('intents2.json').read())
words = pickle.load(open('words2.pkl', 'rb'))
classes = pickle.load(open('classes2.pkl', 'rb'))

#funkcija za čišćenje rečenice koje je unio korisnik
def clean_up_sentence(sentence):
    ignore_letters = ['!', '?', ',', '.', 'I', 'you', 'me', 'a', 'an']
    #tokenizacija rečenice i lematizacija
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower(), pos='v') for word in sentence_words if word not in ignore_letters]
    print(sentence_words)
    return sentence_words #vraća se očišćena rečenica


# stvaranje vreće riječi za očišćenu rečenicu
def bag_of_words(sentence, words, show_details=True):
    # pozivanje funkcije za čišćenje rečenice
    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return (np.array(bag))

# funkcija iz ChatBox datoteke
def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words, show_details=False) #kao argumente uzima korisnikovu rečenicu i datoteku words
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

    if ints==[]: #ako model za nijednu oznaku nije imao preko 85%
        result = random.choice(intents['intents'][4]['responses']) # izaberi bilo koji odgovor iz oznake "no_answer"
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