# app building library
import streamlit as st

# dataframe libraries
import numpy as np
import pandas as pd

# model libraries
import gensim
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess

# miscellany
import pickle
import gzip
import string

# load vector model
model = Doc2Vec.load('data/doc2vec.model')

# load dataframe
with gzip.open('data/poetry_rec_system.pkl', 'rb') as hello:
    df = pickle.load(hello)

# print recommended poems
def poem_printout(similar_poems):
    
    '''Input a list of tuples of document tags (corresponding to dataframe index values) and percentage of similarity.
       Output a formatted list of poem titles with corresponding poet, link, and percent match.'''
    
    # message from PO-REC
    st.markdown(f'## Great news! I found {len(similar_poems)} poems.')
    # separation line
    st.markdown('-'*75)
    
    # loop over list of tuples
    for i, pct in similar_poems:
        # similarity of match as a percent
        st.markdown(f'### {round(pct*100,1)}% match')
        # title of poem, with corresponding URL to PoetryFoundation.org page, and poet name
        st.markdown(f'[{df.loc[i,"title"].upper()}]({df.loc[i,"poem_url"]}) by {df.loc[i,"poet"]}')
        # genre of poem
        st.markdown(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
        # separation line
        st.markdown('-'*75)

                    
# search based on a keyword
def word_similarity(word, df, model, num_poems=5):
                    
    '''Input a single word, as well as dataframe, Doc2Vec model, and number of poems to recommend (default is 5).
       Output a list of tuples with document tag and percent match, to be fed into poem_printout function.'''
                    
    # find the vector for that word, if it exists within the model
    vec = model[word]
                    
    # find and return poems that are most similar to that word vector
    similar_poems = model.docvecs.most_similar([vec], topn=num_poems)
    return similar_poems


# search based on a phrase
def phrase_similarity(phrase, df, model, num_poems=5):
                    
    '''Input a text of any length, as well as dataframe, Doc2Vec model, and number of poems to recommend (default is 5).
       Output a list of tuples with document tag and percent match, to be fed into poem_printout function.'''
                    
    # process the input in the same manner of documents in the model
    words = simple_preprocess(phrase)
                    
    # create a vector for the input text based on the model
    vec = model.infer_vector(words)
    
    # find and return poems that are most similar to that word vector                
    similar_poems = model.docvecs.most_similar([vec], topn=num_poems)
    return similar_poems

                    
# search based on a poem in the dataframe              
def poem_similarity(title, poet, df, model, num_poems=5):
                    
    '''Input a title of a poem and its corresponding poet, as well as dataframe, Doc2Vec model, and number 
       of poems to recommend (default is 5).
       Output a list of tuples with document tag and percent match, to be fed into poem_printout function.'''
                    
    # find the index value for the poem
    # NOTE: since some poems have the same title but different poets, both fields are required
    poem_id = df[(df.title == title) & (df.poet == poet)].index[0]
    
    # find the vector for that poem
    # NOTE: index value and document tag should be the same
    vector = model.docvecs[poem_id]
    
    # find and return poems that are most similar to that word vector
    # add one to the input number of poems and slice off first result because the first result will always be
    # the same as the input poem
    similar_poems = model.docvecs.most_similar([vector], topn=num_poems+1)[1:]
    return similar_poems

                    
# filter recommended poems based on various parameters
def poem_filter(similar_poems, df, genre=None, min_lines=None, max_lines=None, min_len_line=None, max_len_line=None,
                polarity=None, end_rhyme=None):
                    
    '''Input a list of tuples of document tags (corresponding to dataframe index values) and percentage of similarity,
       as well as a dataframe and any desired parameters.
       
       Genre: one of ['beat', 'black_arts_movement', 'black_mountain', 'confessional', 'harlem_renaissance', 'imagist',
                      'language_poetry', 'modern', 'new_york_school', 'new_york_school_2nd_generation', 'objectivist',
                      'romantic', 'victorian']
       min_lines: minimum number of lines a poem must have to be considered
       max_lines: maximum number of lines a poem must have to be considered
       min_len_line: minimum average of words in each line of a poem to be considered
       max_len_line: minimum average of words in each line of a poem to be considered
       polarity: one of ['positive', 'neutral', 'negative']
       end_rhyme: one of [0, 1], 0 being poems with few to no end rhymes and 1 being a poem with end rhymes
       
       Output a list of poems that satisfy the filter.'''
    
    # genre filter
    if genre:
        # limit dataframe to poems within input genre
        df = df[df.genre == genre]
    
    # poem length filter
    if min_lines:
        if max_lines:
            # if user inputs both values
            df = df[(df.num_lines >= min_lines) & (df.num_lines <= max_lines)]
                
        else:
            # if user only inputs minimum length of poem
            df = df[df.num_lines >= min_lines]
                    
    # if user only inputs maximum length of poem                
    elif max_lines:
        df = df[df.num_lines <= max_lines]
    
    # line length filter                
    if min_len_line:
        if max_len_line:
            # if user inputs both values
            df = df[(df.avg_len_line >= min_len_line) & (df.avg_len_line <= max_len_line)]
        else:
            # if user only inputs minimum length of line
            df = df[df.avg_len_line >= min_len_line]
                    
    # if user only inputs minimum length of line
    elif max_len_line:
        df = df[df.avg_len_line <= max_len_line]
    
    # sentiment filter                
    if polarity:
        # limit dataframe to poems within input sentiment polarity
        df = df[df.sentiment_polarity == polarity]
        
    # end rhyme filter
    # limit dataframe to poems within that end_rhyme value
    if end_rhyme == False:
        df = df[df.end_rhyme == 0]
    elif end_rhyme == True:
        df = df[df.end_rhyme == 1]
    
    # re-create the original list using only poems that satisfy the filters (i.e. appear in the filtered dataframe)
    similar_poems = [(i, pct) for i, pct in similar_poems if i in df.index]
    
    # return poems if available
    if similar_poems:
        return similar_poems
                    
    # return a message if the list is empty
    else:
        print('Filter too fine. Please retry.')                    

                    
# filter recommended poems based on various parameters
def filter_process(similar_poems, df):
                    
    '''Input a list of tuples of document tags (corresponding to dataframe index values) and percentage of similarity,
       as well as a dataframe and any desired parameters.
       Output is a printout of filter list if filter is desired or PO-REC input list if no filter desired.'''
                    
    # a fateful decision, to filter or not to filter
    to_filter_option = st.radio(
        'Would you like to use my filter?',
        ['no', 'yes'])

    # filter parameters in the sidebar
    if to_filter_option == 'yes':
        # title
        st.sidebar.markdown('## FILTER OPTIONS')

        # instantiate a blank option and add all genres from dataframe as other options
        genres = ['']
        genres.extend(df['genre'].unique())
        # reformat genre titles
        genres = [genre.replace('_', ' ').title() for genre in genres]
                    
        # selected genre (if selected), formatted back to be used in dataframe
        genre_option = st.sidebar.selectbox(
            'Pick a genre:',
             genres).replace(' ', '_').lower()

        # range slider with minimum and maximum poem length values
        num_lines_options = st.sidebar.slider(
            'Select a range for the number of lines in each recommendation.',
            # use dataframe's min and max values as ends to slider
            int(df.num_lines.min()), int(df.num_lines.max()),
            (int(df.num_lines.min()), int(df.num_lines.max())))

        # range slider with minimum and maximum line length values
        len_line_options = st.sidebar.slider(
            "Select a range for the average number of words per line in each recommendation.",
            # use dataframe's min and max values as ends to slider
            df.avg_len_line.min(), df.avg_len_line.max(), 
            (df.avg_len_line.min(), df.avg_len_line.max()))
        
        # instantiate a blank option and add polarities from dataframe as other options
        polarities = ['']
        polarities.extend(df['sentiment_polarity'].unique())
        
        # selected sentiment (if selected)
        polarity_option = st.sidebar.selectbox(
            'Pick a sentiment:',
             polarities)

        # selected end rhymes (if selected)
        end_rhymes_option = st.sidebar.radio(
        'Do you want poems with end rhymes?',
        # list with blank option, no, or yes
        ['', 'no', 'yes'])
        if end_rhymes_option:
            # convert no to 0
            if end_rhymes_option == 'no':
                end_rhymes_option = 0
            # convert yes to 1
            else:
                end_rhymes_option = 1

        # run poem_filter function from above
        similar_poems = poem_filter(similar_poems, df,
                                    genre=genre_option,
                                    min_lines=num_lines_options[0],
                                    max_lines=num_lines_options[1],
                                    min_len_line=len_line_options[0],
                                    max_len_line=len_line_options[1],
                                    polarity=polarity_option,
                                    end_rhyme=end_rhymes_option)

        # if one or more poems pass through the filter, print them out
        if similar_poems:
            poem_printout(similar_poems)
        # if no poems pass through the filter, message the user
        else:
            st.markdown('#### Filter too fine. Please retry.')

    # if no filter desired, print out the recommendations
    else:
        poem_printout(similar_poems)