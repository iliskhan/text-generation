import os
import re

import numpy as np 
import pandas as pd 

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation


path_to_data = '../data/raw_data/comments.json'

mystem = Mystem() 

russian_stopwords = stopwords.words("russian")

def main():

    data = pd.read_json(path_to_data, encoding='utf8')

    #удаление постов с пустыми комментами
    data = data[data['комменты'].str.len() != 0]
    
    comments = data['комменты']
    for i in comments:
        for j in i:
            print(preprocess_text(j))
        break

def preprocess_text(text):

    tokens = re.sub(r'[^а-я\s]', '', text.lower())

    if tokens.replace(' ','').isalpha():

        tokens = mystem.lemmatize(tokens)

        tokens = [token for token in tokens if token not in russian_stopwords\
                  and token != " " and token != "\n"]
        
        text = " ".join(tokens)
        
        return text


if __name__ == '__main__':
    main()