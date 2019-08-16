import os
import re

import numpy as np 
import pandas as pd 

from string import punctuation
from pymystem3 import Mystem
from nltk.corpus import stopwords
from multiprocessing import Pool


path_to_data = '../data/raw_data/comments.json'
destination_path = '../data/preprocessed_data/comments.json'

mystem = Mystem() 

russian_stopwords = stopwords.words("russian")

def main():

    data = pd.read_json(path_to_data, encoding='utf8')

    #удаление постов с пустыми комментами
    data = data[data['комменты'].str.len() != 0]
    
    comments = data['комменты']
    descriptions = data['описание']

    cleared_descriptions = []
    cleared_comments = []

    with Pool() as p:

        result = p.starmap(preprocess_data, zip(descriptions, comments))
    
    [cleared_descriptions.append(i[0]) for i in result]
    [cleared_comments.append(i[1]) for i in result]            

    # for description, comments_list in zip(descriptions, comments):

    #     description, comments_list = preprocess_data(description, comments_list)


    cleared_data = pd.DataFrame()

    cleared_data['описания'] = cleared_descriptions
    cleared_data['комменты'] = cleared_comments

    with open(destination_path, 'w', encoding='utf8') as f:
        cleared_data.to_json(f, force_ascii=False)

def preprocess_data(description, comments):

    cleared_comments = []
    cleared_description = preprocess_text(description)

    for comment in comments:

        comment = preprocess_text(comment)
        if comment is not None:
            cleared_comments.append(comment)

    return cleared_description, cleared_comments


def preprocess_text(text):

    #удаление разрывов строк
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')

    #удаление хештегов
    tokens = re.sub(r'[#]\w+','', text.lower())
    
    #удаление всех символов кроме букв и пробелов
    tokens = re.sub(r'[^а-я\s]', '', tokens)

    tokens = re.sub(r'(\s)\1+', r'\1', tokens)

    if tokens.replace(' ','').isalpha():
        tokens = mystem.lemmatize(tokens)

        tokens = [token for token in tokens if token not in russian_stopwords
                  and token != " " and token != "\n"]
        
        text = " ".join(tokens)

        text = text.replace('\n', '')
        text = re.sub(r'(\s)\1+', r'\1', text)
        
        return text


if __name__ == '__main__':
    main()