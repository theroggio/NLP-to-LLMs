############################
# NAMED ENTITY RECOGNITION #
# using CRF model          #
###########################

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_classification_report

# reading the dataset
df = pd.read_csv("data/ner_dataset.csv", encoding="ISO-8859-1")

# if there are ANY null entries in ANY column
if df.isnull().any().any():
    # fill them up from last correct observation
    df = df.fillna(method = 'ffill')

# This is a class te get sentence. The each sentence will be list of tuples with its tag and pos.
class sentence(object):
    def __init__(self, df):
        self.n_sent = 1
        self.df = df
        self.empty = False
        agg = lambda s : [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                       s['POS'].values.tolist(),
                                                       s['Tag'].values.tolist())]
        self.grouped = self.df.groupby("Sentence #").apply(agg)
        self.sentences = [s for s in self.grouped]
        
    def get_text(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent +=1
            return s
        except:
            return None

# we define the getter (class just defined) over our dataset df
getter = sentence(df)
# we try it out on one sentence
sent = getter.get_text()
print("Here is one sentence extracted from the dataset:\n")
print(sent)

# this gets all the sentences
sentences = getter.sentences

# function to extract featurs
def word2features(sent, i):

    word = sent[i][0]
    postag = sent[i][1]

    # bias is fixed to 1.0
    # we take (i) lowercase word, (ii)-(iii) endings, (iv)-(vi) if it has capitals, titles or digits, (vii)-(viii) postag and postag category
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        # update the features based on previous word if possible
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        # update the features based on sequent word if possible
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

# functions to return all features, labels and tokens 
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

# this create the real dataset: X = all features, Y = all labels
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

# we split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# create the model
crf = CRF(algorithm = 'lbfgs',
         c1 = 0.1,
         c2 = 0.1,
         max_iterations = 100,
         all_possible_transitions = False)
# fit the model
crf.fit(X_train, y_train)
# predict over the tests
y_pred = crf.predict(X_test)

# compute the F1-score over predictions
f1_score = flat_f1_score(y_test, y_pred, average = 'weighted')
print("F1-SCORE:\n")
print(f1_score)

# compute the whole classification report
report = flat_classification_report(y_test, y_pred)
print("REPORT:\n")
print(report)


