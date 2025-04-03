## exercise to understand bag of words

print("MANUAL WAY\n")
import nltk
# Download stopwords and tokenizer if you haven't already
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Import the regular expressions module to help with text processing
import re  
# import default dict to create the vocabulary
from collections import (
    defaultdict,
)  

# Sample corpus of text - a small dataset of sentences to analyze
corpus = [
    "Tokenization is the process of breaking text into words.",
    "Vocabulary is the collection of unique words.",
    "The process of tokenizing is essential in NLP.",
]

# Initialize a defaultdict with integer values to store word frequencies
# defaultdict(int) initializes each new key with a default integer value of 0
vocab = defaultdict(int)

# Loop through each sentence in the corpus to tokenize and normalize
for sentence in corpus:
    # Use regular expressions to find words composed of alphanumeric characters only
    words = re.findall(r"\b\w+\b", sentence.lower())
    # For each word found, increment its count in the vocab dictionary
    for word in words:
        vocab[word] += 1

# Convert the defaultdict vocab to a regular dictionary for easier handling and sorting
# Sort the dictionary by word frequency in descending order and convert it to a new dictionary
vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))

# Display the sorted vocabulary with each word and its frequency count
print("Vocabulary with Frequencies:\n", vocab)

# now that you have the vocabulary you can also save each sentence as it is represented by the bag of words
def create_bow_vector(sentence, vocab):
    vector = [0] * len(vocab)  # Initialize a vector of zeros
    # we check every word
    for word in sentence.split():
        # if the word is in the vocabulary
        if word in vocab:
            # take the index, for that we need to have a list
            idx = list(vocab).index(word)  # Find the index of the word in the vocabulary
            vector[idx] += 1  # Increment the count at that index
    return vector

bow_vectors = [create_bow_vector(sentence, vocab) for sentence in corpus]
print("Bow vectors:\n")
print(bow_vectors)


###### standardized way ###########
# it is uncommon nowadays to do this all by hand
# scikit-learn is optimized to do this faster and better

print("\n\nSCIKIT WAY\n")
from sklearn.feature_extraction.text import CountVectorizer
# Original corpus
corpus = [
    "Python is amazing and fun.",
    "Python is not just fun but also powerful.",
    "Learning Python is fun!",
]
# Create a CountVectorizer Object
vectorizer = CountVectorizer()
# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)
# Print the generated vocabulary
print("Vocabulary:", vectorizer.get_feature_names_out())
# Print the Bag-of-Words matrix
print("\nBoW Representation:")
print(X.toarray())
