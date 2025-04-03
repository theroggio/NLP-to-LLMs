############################
# NAMED ENTITY RECOGNITION #
# using LSTM model          #
###########################

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_classification_report
import torch

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

# parameters
# Number of data points passed in each iteration
batch_size = 64 
# Passes through entire dataset
epochs = 8
# Maximum length of review
max_len = 75 
# Dimension of embedding vector
embedding = 40 

# CREATE DATASET
# first we define all possible words and tags
words = list(df['Word'].unique())
tags = list(df['Tag'].unique())

# define how from one word get its index, we include the pad and unk as first 2
word_2_index = {w : i + 2 for i, w in enumerate(words)}
word_2_index["UNK"] = 1
word_2_index["PAD"] = 0

# we define how access the tags, including the pad tag as 0
tag_2_index = {t : i + 1 for i, t in enumerate(tags)}
tag_2_index["PAD"] = 0

# now the inverse, from index to word and from index to tag 
idx2word = {i: w for w, i in word_2_index.items()}
idx2tag = {i: w for w, i in tag_2_index.items()}

# Converting each sentence into list of index from list of tokens
X = [ torch.Tensor([word_2_index[w[0]] for w in s]) for s in sentences]
X = [ x[:max_len] for x in X ]
# Padding each sequence to have same length  of each word
X = torch.nn.utils.rnn.pad_sequence(sequences = X, padding_value= word_2_index["PAD"], padding_side='right')

# Convert label to index
y = [ torch.Tensor([tag_2_index[w[2]] for w in s]) for s in sentences]
y = [ x[:max_len] for x in y ]
# padding
y = torch.nn.utils.rnn.pad_sequence(sequences = y, padding_value= tag_2_index["PAD"], padding_side='right')
# One hot encoded labels
num_tag = df['Tag'].nunique()
y = [ torch.nn.functional.one_hot(i.long(), num_classes = num_tag + 1) for i in y]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)


## create the LSTM model

# model
from torchcrf import CRF
class BiLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, max_len):
        super(BiLSTM, self).__init__()
        
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True,
                            bidirectional=True, dropout=0.1)
        self.fc = torch.nn.Linear(hidden_dim * 2, 50)
        self.relu = torch.nn.ReLU()
        self.crf = CRF(num_tags=num_tags+1, batch_first=True)
    
    def forward(self, x, tags=None):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.relu(self.fc(x))
        
        if tags is not None:  # Training mode
            loss = -self.crf(x, tags, reduction='mean')
            return loss
        else:  # Inference mode
            return self.crf.decode(x)

import ipdb; ipdb.set_trace()
model = BiLSTM(len(words)+2, embedding, 50, num_tag, max_len)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=epochs,
                    validation_split=0.1, callbacks=[checkpointer])

model.test(X_test)
