import pickle
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


from keras.models import load_model
model = load_model('chatbot_model.h5')

train_x = pickle.load(open("train_x.pkl", "rb"))
train_y = pickle.load(open('train_y.pkl', 'rb'))
test_x = pickle.load(open('test_x.pkl', 'rb'))
test_y = pickle.load(open('test_y.pkl', 'rb'))

#stvaranje predviđanja
predictions = model.predict(test_x, batch_size=3, verbose=0)
pred = np.argmax(predictions, axis=1) #uzima najveću vrijednost
test_y = np.argmax(test_y, axis=1)


#matrica konfuzije i klasifikacijsko izvješće

labels_name = ['Sport_and_recreation', 'canteens', 'copy_shop', 'doctor', 'fesb_page', 'free time', 'goodbye', 'greeting_1', 'greeting_2', 'library', 'menu', 'options', 'rooms', 'services', 'student_service', 'thanks']

print(classification_report(test_y, pred))
labels = []
for i in range(0 , 16):
    labels.append(i)

cm=confusion_matrix(test_y, pred, labels=labels)
cmd_obj = ConfusionMatrixDisplay(cm, display_labels=labels_name)
cmd_obj.plot()
cmd_obj.ax_.set(
                title='Matrica konfuzije za chatbot',
                xlabel='Predviđene klase',
                ylabel='Točne klase')
plt.show()
