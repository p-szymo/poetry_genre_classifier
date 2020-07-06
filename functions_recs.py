import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess

import pickle
import gzip
import string

model = Doc2Vec.load('data/doc2vec.model')

# uncomment to load
with gzip.open('data/poetry_rec_system.pkl', 'rb') as hello:
    df = pickle.load(hello)

def poem_printout(similar_poems):
    st.markdown(f'## Great news! I found {len(similar_poems)} poems.')
    st.markdown('-'*75)
    for i, pct in similar_poems:
        st.markdown(f'### {round(pct*100,1)}% match')
        st.markdown(f'[{df.loc[i,"title"].upper()}]({df.loc[i,"poem_url"]}) by {df.loc[i,"poet"]}')
        st.markdown(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
        st.markdown('-'*75)

def word_similarity(word, df, model, num_poems=5, to_print=True):
    # search based on a keyword
    vec = model[word]
    similar_poems = model.docvecs.most_similar([vec], topn=num_poems)
    # if to_print:
    #     poem_printout(similar_poems)
    return similar_poems



def text_similarity(text, df, model, num_poems=5, to_print=True):
    # search based on a string
    words = simple_preprocess(text)
    vec = model.infer_vector(words)
    similar_poems = model.docvecs.most_similar([vec], topn=num_poems)
    # if to_print:
    #     poem_printout(similar_poems)
    return similar_poems

              
def poem_similarity(title, poet, df, model, num_poems=5, to_print=True):
    poem_id = df[(df.title == title) & (df.poet == poet)].index[0]
    poem = df.loc[poem_id, 'string_titled']
    poem_words = poem.translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits))\
                     .strip().lower().split()
    vector = model.docvecs[poem_id]
    similar_poems = model.docvecs.most_similar([vector], topn=num_poems+1)[1:]
    # if to_print:
    #     poem_printout(similar_poems)
    return similar_poems


def poem_filter(similar_poems, df, genre=None, min_lines=None, max_lines=None, min_len_line=None, max_len_line=None,
                polarity=None, end_rhyme=None, to_print=True):
    
    if genre:
        df = df[df.genre == genre]
    
    if min_lines:
        if max_lines:
            df = df[(df.num_lines >= min_lines) & (df.num_lines <= max_lines)]
        else:
            df = df[df.num_lines >= min_lines]
    elif max_lines:
        df = df[df.num_lines <= max_lines]
    
    if min_len_line:
        if max_len_line:
            df = df[(df.avg_len_line >= min_len_line) & (df.avg_len_line <= max_len_line)]
        else:
            df = df[df.avg_len_line >= min_len_line]
    elif max_len_line:
        df = df[df.avg_len_line <= max_len_line]
        
    if polarity:
        df = df[df.sentiment_polarity == polarity]
        
    if end_rhyme == False:
        df = df[df.end_rhyme == 0]
    elif end_rhyme == True:
        df = df[df.end_rhyme == 1]
    
    similar_poems = [(i, pct) for i, pct in similar_poems if i in df.index]
    
    if to_print:
        for i, pct in similar_poems:
            print(f'{round(pct*100,1)}% match')
            print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
            print(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
            print(f'URL: {df.loc[i,"poem_url"]}')
            print('-'*100)
    
    if similar_poems:
        return similar_poems
    else:
        print('Filter too fine. Please retry.')


def filter_process(similar_poems, df):
    to_filter_option = st.radio(
        'Would you like to filter?',
        ['no', 'yes'])

    if to_filter_option == 'yes':
        st.sidebar.markdown('## FILTER OPTIONS')

        genres = ['']
        genres.extend(df['genre'].unique())
        genres = [genre.replace('_', ' ').title() for genre in genres]
        genre_option = st.sidebar.selectbox(
            'Pick a genre:',
             genres).replace(' ', '_').lower()

        num_lines_options = st.sidebar.slider(
            'Select a range for the number of lines in each recommendation.',
            int(df.num_lines.min()), int(df.num_lines.max()),
            (int(df.num_lines.min()), int(df.num_lines.max())))

        len_line_options = st.sidebar.slider(
            "Select a range for the average number of words per line in each recommendation.",
            df.avg_len_line.min(), df.avg_len_line.max(), 
            (df.avg_len_line.min(), df.avg_len_line.max()))
        
        polarities = ['']
        polarities.extend(df['sentiment_polarity'].unique())
        polarity_option = st.sidebar.selectbox(
            'Pick a sentiment:',
             polarities)

        end_rhymes_option = st.sidebar.radio(
        'Do you want poems with end rhymes?',
        ['', 'no', 'yes'])
        if end_rhymes_option:
            if end_rhymes_option == 'no':
                end_rhymes_option = 0
            else:
                end_rhymes_option = 1

        similar_poems = poem_filter(similar_poems, df,
                                    genre=genre_option,
                                    min_lines=num_lines_options[0],
                                    max_lines=num_lines_options[1],
                                    min_len_line=len_line_options[0],
                                    max_len_line=len_line_options[1],
                                    polarity=polarity_option,
                                    end_rhyme=end_rhymes_option)

        if similar_poems:
            poem_printout(similar_poems)
        else:
            st.markdown('#### Filter too fine. Please retry.')

    else:
        poem_printout(similar_poems)



def word_similarity_jup(word, df, model, num_poems=5, to_print=True):
    # search based on a keyword
    vec = model[word]
    similar_poems = model.docvecs.most_similar([vec], topn=num_poems)
    if to_print:
        for i, pct in similar_poems:
            print(f'{round(pct*100,1)}% match')
            print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
            print(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
            print(f'URL: {df.loc[i,"poem_url"]}')
            print('-'*100)
    return similar_poems   
              
def text_similarity_jup(text, df, model, num_poems=5, to_print=True):
    # search based on a string
    words = text.translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits))\
                     .strip().lower().split()
    vector = model.infer_vector(words)
    similar_poems = model.docvecs.most_similar([vector], topn=num_poems)
    if to_print:
        for i, pct in similar_poems:
            print(f'{round(pct*100,1)}% match')
            print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
            print(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
            print(f'URL: {df.loc[i,"poem_url"]}')
            print('-'*100)
    return similar_poems          
              
def poem_similarity_jup(title, df, model, num_poems=5, to_print=True):
    poem_id = df[df['title'] == title].index[0]
    poem = df.loc[poem_id, 'string_titled']
    poem_words = poem.translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits))\
                     .strip().lower().split()
    vector = model.docvecs[poem_id]
    similar_poems = model.docvecs.most_similar([vector], topn=num_poems+1)[1:]
    if to_print:
        for i, pct in similar_poems:
            print(f'{round(pct*100,1)}% match')
            print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
            print(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
            print(f'URL: {df.loc[i,"poem_url"]}')
            print('-'*100)
    return similar_poems
                  
def poem_filter_jup(similar_poems, df, genre=None, min_lines=None, max_lines=None, min_len_line=None, max_len_line=None,
                polarity=None, end_rhyme=None, to_print=True
#                 min_syllables=None, max_syllables=None
                       ):
    
    if genre:
        df = df[df.genre == genre]
    
    if min_lines:
        if max_lines:
            df = df[(df.num_lines >= min_lines) & (df.num_lines <= max_lines)]
        else:
            df = df[df.num_lines >= min_lines]
    elif max_lines:
        df = df[df.num_lines <= max_lines]
    
    if min_len_line:
        if max_len_line:
            df = df[(df.avg_len_line >= min_len_line) & (df.avg_len_line <= max_len_line)]
        else:
            df = df[df.avg_len_line >= min_len_line]
    elif max_len_line:
        df = df[df.avg_len_line <= max_len_line]
        
    if polarity:
        df = df[df.sentiment_polarity == polarity]
        
    if end_rhyme == False:
        df = df[df.end_rhyme == 0]
    elif end_rhyme == True:
        df = df[df.end_rhyme == 1]
    
    similar_poems = [(i, pct) for i, pct in similar_poems if i in df.index]
    
    if to_print:
        for i, pct in similar_poems:
            print(f'{round(pct*100,1)}% match')
            print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
            print(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
            print(f'URL: {df.loc[i,"poem_url"]}')
            print('-'*100)
    
    if similar_poems:
        return similar_poems
    else:
        print('Filter too fine. Please retry.')
                  
                  
def make_tsne_subset_jup(tsne_df: pd.DataFrame, poetry_df: pd.DataFrame, column: str, genre: str) -> pd.DataFrame:
    """
    Takes a dataframe of a fit/transformed t-SNE object. Beer ID/Doc Tags are the index.
    Second argument is a string that is the style of beer (as per stated style in the beers dataset)
    Returns a Pandas Dataframe with t-SNE coordinates of that specific style, allowing you to show clustering
    """
    subset = poetry_df.loc[poetry_df[column] == genre]
    subset_set = set(subset.index)
    match = set(tsne_df.index).intersection(subset_set)
    style_subset = tsne_df[tsne_df.index.isin(match)]
    return style_subset