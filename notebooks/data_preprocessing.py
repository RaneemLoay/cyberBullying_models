import pandas as pd
import nltk
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import numpy as np
import random
from googletrans import Translator

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
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        u"\U0001F926-\U0001F937"  # Supplemental symbols
        u"\U0000203C-\U0000203D"  # Special symbols
        u"\U00002500-\U00002BEF"  # Various symbols
        u"\U0000FE00-\U0000FE0F"  # Variation Selectors
        u"\U0001FB00-\U0001FBFF"  # Symbols and Pictographs Extended-B
        u"\U0001F1E6-\U0001F1FF"  # Flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

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
# Synonym replacement using a simple list (a more sophisticated approach would use word embeddings or a thesaurus)
synonyms = {
    "سعيد": ["فرحان", "مسرور", "مبتهج", "مبسوط"],
    "غاضب": ["زعلان", "مستاء", "ساخط", "مغتاظ"],
    "جميل": ["وسيم", "حسن", "بديع", "رائع"],
    "قبيح": ["بشع", "سيء", "رديء", "مشوه"],
    "سريع": ["مسرع", "عاجل", "خاطف", "مُلهِم"],
    "بطيء": ["متمهل", "متأني", "متباطئ", "ثقيل"],
    "كبير": ["ضخم", "عظيم", "هائل", "جسيم"],
    "صغير": ["قليل", "ضئيل", "قزم", "مصغر"],
    "ذكي": ["نبيه", "فطن", "حاذق", "عبقري"],
    "غبي": ["أحمق", "بليد", "غافل", "جاهل"],
    "مهم": ["ضروري", "أساسي", "حيوي", "جوهري"],
    "مثير": ["شيق", "جذاب", "مبهر", "ممتع"],
    "متعب": ["مرهق", "مجهد", "مضني", "مكافح", "تعبان"],
    "هادئ": ["رائق", "سكوني", "طمأنينة", "مستقر"],
    "خائف": ["مرعوب", "مذعور", "مخيف", "قلق"],
    "قوي": ["صلب", "متين", "قاسي", "صارم"],
    "ضعيف": ["هزيل", "متداع", "مرهف", "متقلب"],
    "بشوش": ["مبتسم", "مسرور", "فرح", "مبتهج"],
    "ممل": ["رتيب", "متعَب", "مضجر", "كئيب"],
    "جديد": ["حديث", "طازج", "مبتكر", "عصري"],
    "قديم": ["عتيق", "تليد", "كلاسيكي", "أثري"],
    "مشهور": ["معروف", "ذائع الصيت", "بارز", "شائع"],
    "مجهول": ["غير معروف", "مغمور", "خفي", "مبهم"],
    "طويل": ["ممتد", "مرتفِع", "طويل القامة", "ممدود"],
    "كثير": ["غزير", "وافر", "فيض", "زاخر"],
    "قليل": ["نادر", "شحيح", "محدود", "ضئيل"],
}


def synonym_replacement(text):
    words = text.split()
    new_words = words.copy()
    for i in range(len(words)):
        if words[i] in synonyms:
            new_words[i] = random.choice(synonyms[words[i]])
    return ' '.join(new_words)

# Back-translation using googletrans library
translator = Translator()

def back_translate(text):
    translated = translator.translate(text, src='ar', dest='en').text
    back_translated = translator.translate(translated, src='en', dest='ar').text
    return back_translated

# Noise injection by adding random characters
def add_noise(text):
    noisy_text = ""
    for char in text:
        if random.uniform(0, 1) < 0.1:
            noisy_text += random.choice("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
        noisy_text += char
    return noisy_text

# Random deletion of words
def random_deletion(text, p=0.2):
    words = text.split()
    if len(words) == 1:
        return text
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return ' '.join(new_words) if new_words else words[0]

# Random swap of words
import random

def random_swap(text, n=1):
    words = text.split()
    if len(words) < 2:
        return text  # No swap is possible if there are fewer than 2 words
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return ' '.join(new_words)

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


