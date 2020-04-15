from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import os
import pickle

def preprocess(data):
    vectorizer = HashingVectorizer(stop_words='english', n_features=2**18, binary=True, norm=None)
    return vectorizer.transform(data)

# Util method for storing the preprocessed data
def store_preprocess(data, filename):
    with open(filename, 'wb') as train_data:
        pickle.dump(data, train_data)

# Util methods for loading data from pickle if a preprocessed file is present, else load and preprocess
def load_training_data():
    y_train = pickle.load(open('Data/sklearn-data.pickle', 'rb'))['y_train']
    if not os.path.exists('Data/train-data.pickle'):
        x_train = preprocess(pickle.load(open('Data/sklearn-data.pickle', 'rb'))['x_train'])
        store_preprocess(x_train, 'Data/train-data.pickle')
    else:
        x_train = pickle.load(open('Data/train-data.pickle', 'rb'))
    return x_train, y_train

def load_test_data():
    y_test = pickle.load(open('Data/sklearn-data.pickle', 'rb'))['y_test']
    if not os.path.exists('Data/test-data.pickle'):
        x_test = preprocess(pickle.load(open('Data/sklearn-data.pickle', 'rb'))['x_test'])
        store_preprocess(x_test, 'Data/test-data.pickle')
    else:
        x_test = pickle.load(open('Data/test-data.pickle', 'rb'))
    return x_test, y_test


# Load data
x_train, y_train = load_training_data()
x_test, y_test = load_test_data()

# Naive Bayes classificiation
classifier = BernoulliNB()

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(f"Naive Bayes accuracy: {accuracy_score(y_test, y_pred)}")

# Decision Tree classifier
classifier = DecisionTreeClassifier(min_samples_leaf=10, min_impurity_decrease=0.01)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(f"Decision Tree accuracy: {accuracy_score(y_test, y_pred)}")