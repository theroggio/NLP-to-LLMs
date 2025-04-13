# all utils and small code components

# Tokenizer
import tiktoken
def get_tokenizer(tokenizer_name, model_name = None):
    if model_name == None:
        enc = tiktoken.get_encoding(tokenizer_name)
    else:
        enc = tiktoken.encoding_for_model(model_name)
    assert enc.decode(enc.encode("Bella fra")) == "Bella fra"
    return enc


