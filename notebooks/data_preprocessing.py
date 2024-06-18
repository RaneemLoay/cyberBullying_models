import nltk
import pandas as pd
# import nltk
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
# from nltk.stem.isri import ISRIStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
# from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')

# Define punctuations and stop words
arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

stop_words = stopwords.words("arabic")
custom_stopwords = ['بعد', 'خلال', 'الرغم', 'بها', 'به', 'بينما', '', '.']
stop_words.extend(custom_stopwords)

arabic_diacritics = re.compile(r"""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

def remove_emoji(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U0001F900-\U0001F9FF"  # emojis
                                        u"\U000025A0-\U000025FF"  # Geometric Shapes
                                        u"\U00002600-\U000027BF"  # emojis
                                        u"\U00002000-\U0000206F"  # General Punctuation
                                        u"\U0001FA70-\U0001FAFF"  # emojis
                                        u"\U0000E000-\U0000F8FF"  # Private Use Area
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)

def preprocess(text):
    if pd.isnull(text):
        return ''
    # Remove punctuations
    text = text.replace('_', ' ')
    text = "".join([char for char in text if char not in punctuations_list])

    # Remove emojis
    text = remove_emoji(text)

    # Remove numbers
    text = re.sub(r"[0123456789٠١٢٣٤٥٦٧٨٩]", '', text)

    # Remove English letters
    text = re.sub(r"[a-zA-Z]", '', text)

    # Remove diacritics
    text = re.sub(arabic_diacritics, '', text)

    # Normalize text
    text = re.sub(r"[ٱإأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"گ", "ك", text)
    text = re.sub(r"ک", "ك", text)
    text = re.sub(r"؏", "ع", text)

    # Remove elongation
    text = re.sub(r'(.)\1+', r"\1\1", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    text = ' '.join([word for word in tokens if word not in stop_words])
    if pd.isnull(text):
        return ''
    return text

def main():
    # Load dataset
    df = pd.read_csv('D:/pyCharm python/cyberBullyingGP/data/cyber_bullying(11000).csv')
    textBody = df['body']
    label = df['label']

    print(df.head())

    clean_text = []
    i = 0
    for text, lbl in zip(textBody, label):
        temp = preprocess(text)
        if(temp == '') :
            continue
        clean_text.append([temp, lbl.strip()])
        print(clean_text[i])
        print(i)
        i = i + 1

    data = pd.DataFrame(clean_text, columns=['text', 'label'])

    # Check for nulls and duplicates
    print(data.isnull().sum())
    print('Data info:', data.info())
    print('Duplicates:', data.duplicated().sum())
    print('Data shape:', data.shape)

    # Drop duplicates and null values
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    print('Duplicates after cleaning:', data.duplicated().sum())
    print('Null values after cleaning:', data.isnull().sum())
    print('Label value counts:', data['label'].value_counts())
    print('Data shape after cleaning:', data.shape)

    # Create a mapping of labels to integers
    label_mapping = {'not bullying': 0, 'bullying': 1}
    data['label'] = data['label'].map(label_mapping)

    data.to_csv('../data/preprocessed_data.csv', index=False)

    # Verify the mapping
    print(data['label'].value_counts())

if __name__ == "__main__":
    main()


