import os 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data_path = os.path.join(os.getcwd(),'data')

clean_data = pd.read_json(os.path.join(data_path, 'clean_data_binarized.json'))


#get rows where the given attribute is null
no_original_language = clean_data[clean_data['tmdb.original_language'].isnull()]
has_original_language = clean_data[clean_data['tmdb.original_language'].notnull()]

#remove rows with missing country value
has_original_language = has_original_language[has_original_language['tmdb.production_countries'].notnull()]

train_data = has_original_language[has_original_language.columns[1+clean_data.columns.get_loc('Western'):]]
train_label = has_original_language['tmdb.original_language']


X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.3, random_state=42)

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)  

y_pred = clf.predict(X_test)
res = pd.DataFrame({'actual':y_test, 'predicted':y_pred.flatten()})


y_pred = list(y_pred)
y_test = list(y_test)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    blub = y_true + y_pred
    classes = set(blub)
   
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots(2000, 2000)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

from sklearn.utils.multiclass import unique_labels
np.set_printoptions(precision=2)

class_names = y_test
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')

plt.show()