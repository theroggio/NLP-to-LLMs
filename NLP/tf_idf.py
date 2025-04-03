## TD-IDF is an extention of bag of words
# td stays for term frequency, easily how many times the word appear / number of total words
# idf is the inverse document frequency, so number of documents in the corpus / number of documents in which the word appear
# the final weight is TD * IDF 

from sklearn.feature_extraction.text import TfidfVectorizer
# Sample corpus
corpus = [
    "Python is amazing and fun.",
    "Python is not just fun but also powerful.",
    "Learning Python is fun!",
]

# Create the Tf-idf vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the corpus
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# Show the Vocabulary
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())

# Show the TF-IDF Matrix
print("TF-IDF Representation:")
print(X_tfidf.toarray())
