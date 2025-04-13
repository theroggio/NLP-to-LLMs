# import libraries
import pandas as pd
import string
import re
from textblob import TextBlob
import torch
import nltk
from collections import (
    defaultdict,
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# download nltk corpus (first time only)
import nltk
#nltk.download('all')

# Load the amazon review dataset
# this dataset has two keys: 'ReviewText' and 'Positive'
# ReviewText is a string which contains the text of the review
# Positive is an integer, 0 = Negative, 1 = Positive 
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

# function to pre-process the text 
def preprocess_text(text):

    # 1. lower 
    processed_text = text.lower()
    
    # 2. remove html tags here is not needed
    # 3. remove urls is not needed
    # 4. remove punctuation 
    processed_text = processed_text.translate(str.maketrans("","",string.punctuation)) 

    # 5. slang terms, we skip it here, mainly because we would need a very long definition. 
    # BE AWARE: in a real case scanario THIS WOULD BE VERY IMPORTANT, reviews often have slang terms inside
    # 6. spelling correction 
    processed_text = TextBlob(processed_text).correct().string

    # 8. emojis are already checked in this dataset
    # 9. tokenize 
    tokens = word_tokenize(processed_text)

    # 7. remove stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # 10. for brevity we do not stem, directly use lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# apply the function df
df['reviewText'] = df['reviewText'].apply(preprocess_text)

# create the Analyzer
class SentimentAnalyzer(object):
    def __init__(self, df):
        self.bow = defaultdict(int)
        sentences = df['reviewText']
        sentiment = df['Positive']
        for sentence in sentences:
            # Use regular expressions to find words composed of alphanumeric characters only
            words = re.findall(r"\b\w+\b", sentence)
            # For each word found, increment its count in the vocab dictionary
            for word in words:
                self.bow[word] += 1
        self.bowlist = list(self.bow)

    def train(self, df_train):
        self.positive_prob = torch.zeros(len(self.bow), dtype=torch.float)
        self.count = torch.zeros(len(self.bow), dtype=torch.float)
        sentences = df['reviewText']
        sentiments = (df['Positive'] * 2) - 1
        for _id, sentence in enumerate(sentences):
            words = re.findall(r"\b\w+\b", sentence)
            sentiment = sentiments[_id] 
            for word in words:
                try:
                    self.positive_prob[ self.bowlist.index(word) ] += (float)(sentiment)
                    self.count[  self.bowlist.index(word) ] += 1
                except:
                    continue
        self.positive_prob /= self.count
            
    def score(self, text):
        print(text)
        words = re.findall(r"\b\w+\b", text)
        score = 1.0
        for word in words:
            try:
                val = self.positive_prob[ self.bowlist.index(word) ]
                score *= val
            except:
                print("Missing word.")
        print(score > 0)


analyzer = SentimentAnalyzer(df[:1000])
analyzer.train(df[:1000])
analyzer.score(df["reviewText"][1001])

