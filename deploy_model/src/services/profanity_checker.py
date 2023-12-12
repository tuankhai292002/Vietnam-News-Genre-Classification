import gensim  # NLP library
from utils.config import setting
from underthesea import text_normalize, word_tokenize

with open(setting.BAD_WORD_FILE_PATH, 'r', encoding='utf-8') as file:
    BAD_WORD_LIST = file.read().splitlines()

# Get bad words from the text
def get_bad_words(word_list):  
    bad_words_list = []
    for word in word_list:
        if word in BAD_WORD_LIST:
            bad_words_list.append(word)
    return bad_words_list

# Word segment for the text
def segment(text):
    result_list = gensim.utils.simple_preprocess(text)  # Remove special characters and punctuation
    text = ' '.join(result_list)  # Join the elements of the list into a string
    word_list = word_tokenize(text,fixed_words= BAD_WORD_LIST)
    return word_list


# Check if text has bad words
def does_have_bad_words(text):
    text=text_normalize(text)
    word_list = segment(text)
    print(f"segment result: {word_list}")
    bad_word_list = get_bad_words(word_list)
    print(f"bad word: {bad_word_list}")
    return bad_word_list
