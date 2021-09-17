import pickle
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

#učitavanje modela
from keras.models import load_model
model = load_model('chatbot_model1.h5')

#učitavanje podataka za testiranje

test_x = pickle.load(open('test_x1.pkl', 'rb'))
test_y = pickle.load(open('test_y1.pkl', 'rb'))

#stvaranje predviđanja
predictions = model.predict(test_x, batch_size=3, verbose=0)
pred = np.argmax(predictions, axis=1) #uzima najveću vrijednost
test_y = np.argmax(test_y, axis=1)

#definiranje imena za klase
labels_name = ['Sport_and_recreation1',
                'canteens',
                'copy_shop',
                'doctor',
                'doctor_page',
                'elearning_page',
                'fesb_page',
                'free time',
                'goodbye',
                'greeting_1',
                'greeting_2',
                'library1',
                'library2',
                'library_page',
                'mail_page',
                'menu',
                'options',
                'rooms',
                'schedule_page',
                'services',
                'sport_and_recreation2',
                'sport_and_recreation_page',
                'student_service1',
                'student_service2',
                'student_service3',
                'student_service4',
                'student_service_page',
                'thanks']

#stvaranje matrice konfuzije
labels = []
for i in range(0, 27):
    labels.append(i)

cm = confusion_matrix(test_y, pred, labels= labels)
cmd_obj = ConfusionMatrixDisplay(cm)
cmd_obj.plot()
cmd_obj.ax_.set(
                title='Matrica konfuzije za chatbot',
                xlabel='Predviđene oznake',
                ylabel='Točne oznake',
                )
plt.show()

# stvaranje klasifikacijskog izvješća
def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = labels #AAAAAAA
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd, zero_division="warn")).T
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

plt.show()

#histogram za classification report
plt.rcdefaults()
fig, ax = plt.subplots(2,1)
precision, recall, fscore, support = precision_recall_fscore_support(test_y, pred, zero_division="warn")
y_pos = np.arange(len(precision))

test_y= list(test_y)
test_y.append("avg")
test_y=np.array(test_y)
sorted(list(set(test_y)))



performance = precision
ax[0].barh(y_pos, performance, align='edge')
ax[0].set_yticks(y_pos)
ax[0].set_yticklabels(list(set(test_y)))
ax[0].invert_yaxis()  # labels read top-to-bottom
ax[0].set_title('Precision')



performance = recall
ax[1].barh(y_pos, performance, align='edge')
ax[1].set_yticks(y_pos)
ax[1].set_yticklabels(list(set(test_y)))
ax[1].invert_yaxis()  # labels read top-to-bottom
ax[1].set_title('Recall')


plt.rcdefaults()
fig, ax = plt.subplots(figsize =(7, 7))
performance = fscore
ax.barh(y_pos, performance, align='edge')
ax.set_yticks(y_pos)
ax.set_yticklabels(list(set(test_y)))
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Fscore')

plt.show()

