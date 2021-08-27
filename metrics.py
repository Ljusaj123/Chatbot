import pickle
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

#učitavanje modela
from keras.models import load_model
model = load_model('chatbot_model.h5')

#učitavanje podataka za testiranje
train_x = pickle.load(open("train_x.pkl", "rb"))
train_y = pickle.load(open('train_y.pkl', 'rb'))
test_x = pickle.load(open('test_x.pkl', 'rb'))
test_y = pickle.load(open('test_y.pkl', 'rb'))

#stvaranje predviđanja
predictions = model.predict(test_x, batch_size=3, verbose=0)
pred = np.argmax(predictions, axis=1) #uzima najveću vrijednost
test_y = np.argmax(test_y, axis=1)


#definiranje imena za klase
labels_name = ['Sport_and_recreation',
               'canteens', 'copy_shop',
               'doctor',
               'fesb_page',
               'free time',
               'goodbye',
               'greeting_1',
               'greeting_2',
               'library',
               'menu',
               'options',
               'rooms',
               'services',
               'student_service',
               'thanks']
#stvaranje matrice konfuzije
labels = []
for i in range(0, 16):
    labels.append(i)

cm = confusion_matrix(test_y, pred, labels=labels)
cmd_obj = ConfusionMatrixDisplay(cm, display_labels=labels_name)
cmd_obj.plot()
cmd_obj.ax_.set(
                title='Matrica konfuzije za chatbot',
                xlabel='Predviđene klase',
                ylabel='Točne klase',
                )
plt.show()

# stvaranje klasifikacijskog izvješća
def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                cmap='twilight',
                annot=True,
                cbar=True,
                xticklabels=xticks,
                yticklabels=yticks,
                linewidths=.5,
                ax=ax,
                vmin=0.3,
                vmax=2)

plot_classification_report(test_y, pred)


#histogram za classification report
plt.rcdefaults()
fig, ax = plt.subplots(3,1)
precision, recall, fscore, support = precision_recall_fscore_support(test_y, pred)
y_pos = np.arange(len(labels_name))


performance = precision
ax[0].barh(y_pos, performance, align='edge')
ax[0].set_yticks(y_pos)
ax[0].set_yticklabels(labels_name)
ax[0].invert_yaxis()  # labels read top-to-bottom
ax[0].set_title('Precision')



performance = recall
ax[1].barh(y_pos, performance, align='edge')
ax[1].set_yticks(y_pos)
ax[1].set_yticklabels(labels_name)
ax[1].invert_yaxis()  # labels read top-to-bottom
ax[1].set_title('Recall')



performance = fscore
ax[2].barh(y_pos, performance, align='edge')
ax[2].set_yticks(y_pos)
ax[2].set_yticklabels(labels_name)
ax[2].invert_yaxis()  # labels read top-to-bottom
ax[2].set_title('Fscore')

plt.show()
