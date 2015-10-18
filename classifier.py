import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import scipy as sp
    import itertools as it
    import operator as op
    import matplotlib.pyplot as plt

    from sklearn import metrics
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn import cross_validation
    from sklearn.metrics import classification_report


get_neg_ids = lambda x: x[0][1] < 5
get_pos_ids = lambda x: x[0][1] > 5

# gets a list of sentiment scores
get_sent = lambda x: 1 if x[1] > 5 else 0



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    From sklearn's demo pages
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['negative', 'positive'], rotation=45)
    plt.yticks(tick_marks, ['negative', 'positive'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def get_word_set(filename):
    """
    Gets the dictionary from our preprocessed dataset
    """
    with open (filename, 'r') as f:
        tmp = f.readlines()
        tmp = [line.rstrip() for line in tmp]   
    word_split = lambda x: x.split()[2].split(':')[0]
    return set(map(word_split, tmp))


def data_maker(_data):
    """
    hacky method to pivot the data from 'long' to 'wide' for each review.
    Args:
        _data: unzipped group-by'd list of stuff
    Returns:
        list of dictionaries with k: v s.t. k = word, v = count(word). 
    """
    res = []
    for row in _data:
        z = {}
        for line in row:
            k, v = line[1].split(':')
            z[k] = int(v)
        res.append(z)
    return res


def read_file(filename):
    """
    preprocesses some of the weirdness out of the format we saved to
    Returns:
        list of all lines in the files according to: [(id, rating), 'word:<count>']
    """
    with open (filename, 'r') as f:
        tmp = f.readlines()
        tmp = [line.rstrip() for line in tmp]
        split = lambda x: x.split('\t')
        tmp = list(map(split, tmp))
        tup, val = list(zip(*tmp))
        # this bit changes all the string tuples to python tuples. 
        tmp = list(zip(map(eval, tup), val))
    return tmp


def make_train_data(data):
    """ 
    Uses Sklearn's helper method to vectorize a list of dictionary features into a scipy 
    sparse matrix for training
    Returns:
        tuple of data, sparse_matrix
    """
    dv = DictVectorizer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dv.fit_transform(data)




# if __name__ is "__main__":
print("reading file")
raw_train = read_file('./train_mr_output.csv')
raw_test = read_file('./test_mr_output.csv')

print('raw training list: ' + str(len(raw_train)))
print('raw test list: ' + str(len(raw_test)))


# raw_test = read_file('./test_mr_output.csv')
test_grouped = it.groupby(raw_test, op.itemgetter(0))

test_g = [(k, list(g)) for k, g in test_grouped]
test_ids, test_tmp = zip(*test_g)

test = data_maker(test_tmp)


train_grouped = it.groupby(raw_train, op.itemgetter(0))

train_g = [(k, list(g)) for k, g in train_grouped]
train_ids, train_tmp = zip(*train_g)

train = data_maker(train_tmp)

y_test = list(map(get_sent, test_ids))
y_train = list(map(get_sent, train_ids))

data = []
data.extend(train)
data.extend(test)

X = make_train_data(data)

X_train = X[0:25000]
X_test = X[25000:]

nb = MultinomialNB(alpha=0.3)

sgd = SGDClassifier(loss='log',
                    penalty='elasticnet',
                    alpha=0.0001,
                    eta0=0.0001,
                    n_iter=5,
                    warm_start=True)


nb_scores = cross_validation.cross_val_score(nb, X_train, y_train, cv=10)
sgd_scores = cross_validation.cross_val_score(sgd, X_train, y_train, cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (nb_scores.mean(), nb_scores.std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (sgd_scores.mean(), sgd_scores.std() * 2))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nb.fit(X_train, y_train)
    sgd.fit(X_train, y_train)

print(classification_report(nb.predict(X_test), y_test))
print(classification_report(sgd.predict(X_test), y_test))

# from sklearn's example


# Compute confusion matrix
cm = metrics.confusion_matrix(nb.predict(X_test), y_test)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
plt.savefig('basic_cm.pdf')

# Normalize the confusion matrix by row (i.e by the number of samples # in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.savefig('normed_cm.pdf')
