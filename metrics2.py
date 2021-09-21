import pickle
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

#učitavanje modela
from keras.models import load_model
model = load_model('chatbot_model2.h5')


#učitavanje podataka za testiranje

test_x = pickle.load(open('test_x2.pkl', 'rb'))
test_y = pickle.load(open('test_y2.pkl', 'rb'))

#stvaranje predviđanja
predictions = model.predict(test_x, batch_size=8, verbose=0)
pred = np.argmax(predictions, axis=1) #uzima najveću vrijednost
test_y = np.argmax(test_y, axis=1)

#definiranje imena za klase
labels_name = ['Sport_and_recreation',
               'canteens',
               'copy_shop',
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
print(len(labels_name))
#stvaranje matrice konfuzije

labels = []
for i in range(0, 16):
    labels.append(i)
cm = confusion_matrix(test_y, pred, labels=labels)
cmd_obj = ConfusionMatrixDisplay(cm)
cmd_obj.plot()
cmd_obj.ax_.set(
                title='Matrica konfuzije za chatbot',
                xlabel='Predviđene oznake',
                ylabel='Točne oznake',
                )


# stvaranje klasifikacijskog izvješća
def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = labels_name
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

labels_name.remove("avg")
#histogram za klasifikacijsko izvješće
plt.rcdefaults()
fig, ax = plt.subplots(2,1)
precision, recall, fscore, support = precision_recall_fscore_support(test_y, pred)
y_pos = np.arange(len(labels_name))

#preciznost
performance = precision
ax[0].barh(y_pos, performance, align='edge')
ax[0].set_yticks(y_pos)
ax[0].set_yticklabels(labels_name)
ax[0].invert_yaxis()  # labels read top-to-bottom
ax[0].set_title('Precision')


#opoziv
performance = recall
ax[1].barh(y_pos, performance, align='edge')
ax[1].set_yticks(y_pos)
ax[1].set_yticklabels(labels_name)
ax[1].invert_yaxis()  # labels read top-to-bottom
ax[1].set_title('Recall')



#F1 rezultat
plt.rcdefaults()
fig, ax = plt.subplots(figsize =(7, 7))
performance = fscore
ax.barh(y_pos, performance, align='edge')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_name)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Fscore')
plt.show()
