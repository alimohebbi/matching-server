import re
import ssl
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_packages():
    nltk.download('punkt')
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


def remove_punctuation(s):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return s.translate(translator)


def remove_stop_words(input_str):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(input_str)
    result = [i for i in tokens if i not in stop_words]
    joined_result = ' '.join(result)
    if joined_result == '':
        return input_str
    else:
        return joined_result


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatizing(input_str):
    lemmatizer = WordNetLemmatizer()
    input_str = word_tokenize(input_str)
    result = [lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in input_str]
    return ' '.join(result)


def token_camel_case_split(str):
    words = [[str[0]]]
    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
    return ' '.join([''.join(word) for word in words])


def camel_case_split(identifier):
    tokens = word_tokenize(identifier)
    result = []
    for i in tokens:
        result.append(token_camel_case_split(i))
    return ' '.join(result)


def remove_redundant_words(input_str):
    l = input_str.split()
    result = []
    for i in l:
        if input_str.count(i) > 1 and (i not in result) or input_str.count(i) == 1:
            result.append(i)
    return ' '.join(result)


def remove_unusual_char(input_str):
    return re.sub('[^A-Za-z0-9 ]+', '', input_str)


def space_cleaner(input_str):
    return ' '.join(str(input_str).split())


def pre_process(data, train):
    processed_data = data.fillna('')
    processed_data = processed_data.applymap(lambda s: remove_punctuation(s))
    if not train:
        processed_data = processed_data.applymap(lambda s: camel_case_split(s))
    processed_data = processed_data.applymap(lambda s: s.lower())
    processed_data = processed_data.applymap(lambda s: re.sub(r'\d+', '', s))
    processed_data = processed_data.applymap(lambda s: s.strip())
    processed_data = processed_data.applymap(lambda s: space_cleaner(s))
    processed_data = processed_data.applymap(lambda s: remove_stop_words(s))
    processed_data = processed_data.applymap(lambda s: lemmatizing(s))
    if not train:
        processed_data = processed_data.applymap(lambda s: remove_redundant_words(s))
    processed_data = processed_data.applymap(lambda s: remove_unusual_char(s))
    return processed_data


if __name__ == "__main__":
    download_nltk_packages()
    d = {'col1': ['You don\'t have any shopping list', 'input_note	45	Note (Optional) EZ',
                  ' services saved required thinking allows developed',
                  'EZ Tip Calculator is a ssHiBuddy simple tip calculator that allows you to specify the percent you'
                  ' wish to tip'
                  ' and the bill amount. This tip calculator is designed to calculate the tip quickly.']}
    df = pd.DataFrame(data=d)
    data = pre_process(df, False)
    print(data)
