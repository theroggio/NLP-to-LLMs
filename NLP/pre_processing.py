## All pre processing steps required for text data 

text = "This is just an example. I am adding some special characters, punctuation ! and extra weird stuff @companyname :) LMAO\n This is how text you get from the internet looks like! Actually, in html you may have <scipt> such tags </script> or even urls https://my.website.com. \n you often also have speling misatkes and real emojies 😘" 
print("This is your starting text:")
print(text)

# 1. lowercasing 
# Lowercasing makes all capital letters not capital. This is important to have a standardized text. Especially when you use bag-of-words or similar structures where "i" and "I" would be considered different, leading to more memory and a lot of problems since they have the same meaning. 

text = text.lower()
print("\nText is now all lowercase:")
print(text)

# 2. Remove structural HTML tags 
# Language models are oftain trained with data scraped from the web, so it is very common to encounter html tags. However, tags are not part of the "natural language" and thus need to be removed. 

# html tags can be easily matched with regular expressions
import re 

def remove_html_tags(text):
    # define the tags pattern: < + any number of any possible character + >
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

text = remove_html_tags(text)

print("\nText after removing HTML tags:")
print(text)

# 3. Remove URLS 
# As before, webpages often have urls laying around but we do not speak in urls (usually)

def remove_url(text):
    # let's analyze the pattern
    # we can start with http or https, so there is ? after the s
    # \S+ is one or more non space characters
    # www is the expected start of the url, the | before means that either we have www or the match for \S+
    # then we need the "." ! The \ before means that the point is a real point, not a wild card!
    # after the point any sequence of non space characters are the url, it breaks at the first space
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

text = remove_url(text)

print("\nText without URLs:")
print(text)

# 4. Remove punctuation 
# Punctuation does not carry semantic meaning and can introduce noise in the analysis 

# the python class string has already the pattern for punctuation
import string

def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

text = remove_punctuation(text)

print("\nText without punctuation:")
print(text)

# 5. SLANG
# We often use abbreviations, which may be common in spoken language but it is hard for machines to correctly assign them to the sequence of words they stand for. Even NLP is an abbreviation, which in the machine analysis would not match with Natural Language Processing. 
# there is no official corpus of slang terms, especially since they change based on language and period, we add new slang terms each year
# we here define a couple of them for the purpose of demonstrating the functionality 


slang_terms = { 
        "asap": "as soon as possible", 
        "brb" : "be right back", 
        "faq" : "frequently asked questions", 
        "lmao" : "laughing my ass out", 
        "u" : "you" 
    }

def expand_slang(text):
    new_text = []
    for word in text.split():
        if word in slang_terms:
            new_text.append(slang_terms[word])
        else:
            new_text.append(word)
    return " ".join(new_text)

text = expand_slang(text)
print("\nExpanded all slang terms:")
print(text)

# 6. Spelling correction
# correction of spelling mistakes enhance the capabilities of NLP methods, since the machines always assume that the data is "correct". This also ensure that the data is consistent and we are not introducing unwanted noise

# we use a library to handle spelling problems: different languages may need different libraries or parameters
from textblob import TextBlob

text = TextBlob(text).correct().string

print("\nCorrected Text:")
print(text)


# 7. stop words
# stop words are not the words at the end of the sentence, but the common words that ar enot sueful for text udnerstanding, examples: "the" "a" "and" . They appear very frequently but do not help us getting anything out of the sentence. 

# we use ntlk, very famous NLP library!
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# as before, every language has its own!
bad_words = set(stopwords.words("english"))

def remove_stopwords(text):
    new_text = []

    for word in text.split():
        if word in bad_words:
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)

text = remove_stopwords(text)
print("\nText without stopwords:")
print(text)

# 8. EMOJIES
# emojies are actually important, they give us context (for humans even before reading any word), however they pose a real challenge for nlp algorithms since they are non textual. We can, instead of removint them, substitute them with something that preserves their meaning and semantic

# need a library for it 
import emoji

text = emoji.demojize(text)
print("\nText after converting the emojies:")
print(text)


# 9. tokenization 
# here we start with the real processing of data, until now we made the text concie, uniform, standard, but now we need to split it so that it can be fed into our algorithms
# there are different ways to do this, per word, per sentence, many libraries give different options
# we here present one of them 
text = "New text. This is showing how tokenization works! Obviously this should be done in the beginning, otherwise we lose all punctuations. How could we split sentences without punctuation?!"

nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize,sent_tokenize

words_token = word_tokenize(text)
print(f"\nText tokenized for words:\n{words_token}")

sentences_token = sent_tokenize(text)
print(f"\nText tokenized for sentences:\n{sentences_token}")

# 10. Stemming
# i reduces the words to their root, so "walking", "walked" , "walk" all goes to "walk". This also standardize text and reduce the number of different variations we need to store, since they have the same semantic meaning, in this case the walking action 

from nltk.stem.porter import PorterStemmer

def stem_words(text):
    return " ".join([PorterStemmer().stem(word) for word in text.split()])

stemmed_text = stem_words(text)

print(f"\nWe now reduce the words to their roots:\n{stemmed_text}")

# 10: Lemmatization
# this is an advanced form of stemming, where the words are reduced to their lemma, without considering their meaning.
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

text = "I am going to show you some words for which this is different than just stemming, for example is and has would become i and ha using stemming, look what are we getting now!"

# we remove punctuation and take single words
sentence_words = word_tokenize(remove_punctuation(text))

# now for each word we print the lemma 
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word,pos='v')))
