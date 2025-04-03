## All pre processing steps required for text data 

text = "This is just an example. I am adding some special characters, punctuation ! and extra weird stuff @companyname :) LMAO\n This is how text you get from the internet looks like! Actually, in html you may have <scipt> such tags </script> or even urls https://my.website.com. \n you often also have speling misatkes and real emojis 😘" 
print("This is your starting text:")
print(text)

# 1. lowercasing 
# Lowercasing makes all capital letters not capital. This is important to have a standardized text. Especially when you use bag-of-words or similar structures where "i" and "I" would be considered different, leading to more memory and a lot of problems since they have the same meaning. 

text = text.lower()
print("Text is now all lowercase:")
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

print("Text after removing HTML tags:")
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

print("Text without URLs:")
print(text)







