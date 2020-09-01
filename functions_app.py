# app building library
import streamlit as st

# dataframe libraries
import numpy as np
import pandas as pd

# model libraries
import gensim
from gensim.models import Doc2Vec
from functions import simple_process
from sklearn.metrics.pairwise import cosine_similarity

# miscellany
from operator import itemgetter
import pickle
import gzip
import string


# load poetry dataframe
with gzip.open('data/poems_df_rec_system.pkl', 'rb') as hello:
    df = pickle.load(hello)

# load doc2vec dataframe
with gzip.open('data/features_doc2vec_df.pkl', 'rb') as hello:
    df_docvec = pickle.load(hello)

# load doc2vec model
model = Doc2Vec.load('data/doc2vec_final.model')


# print recommended poems
def poem_printout(df, similar_poems):

    '''
    Function to print stylized list of poems.


    Input
    -----
    df : Pandas DataFrame
        Database of poems with at least title, poet,
        genre, and poem URL columns

    similar_poems : list (tup)
        A list of poem indices and percentage of similarity.


    Output
    ------
    Prints a formatted list of poem titles with corresponding
    poet, link, and percent match.

    '''

    # message from PO-REC
    st.markdown(f'## Great news! I found {len(similar_poems)} poems.')
    # separation line
    st.markdown('-'*75)

    # loop over list of tuples
    for i, pct in similar_poems:

        # similarity of match as a percent
        st.markdown(f'### {round(pct*100,1)}% match')

        # title of poem, with corresponding URL to PoetryFoundation.org
        # page, and poet name
        st.markdown(
            f'[{df.loc[i,"title"].upper()}]({df.loc[i,"poem_url"]})\
                by {df.loc[i,"poet"]}')

        # genre of poem
        if df.loc[i, "genre"] != 'new_york_school_2nd_generation':
            st.markdown(f'GENRE: \
                {df.loc[i, "genre"].replace("_", " ").title()}')
        # special case
        else:
            st.markdown('GENRE: New York School 2nd Generation')

        # separation line
        st.markdown('-'*75)


# search based on a keyword
def word_similarity(word, df, model, n=5):

    '''
    Function to find the n-most-similar poems, based on
    an established word vector.


    Input
    -----
    word : str
        Single word whose vector, if known, will be compared
        to document vectors.

    df : Pandas DataFrame
        Database of all poems.

    model : Doc2Vec model
        Fitted Gensim Doc2Vec object.
            `gensim.models.doc2vec.Doc2Vec`


    Optional input
    --------------
    n : int
        The number of poems to return (default=5).


    Output
    ------
    similar_poems : list (tup)
        List of similar poems with poem index as an integer
        and percent similarity as a float.

    '''

    # find the vector for that word, if it exists within
    # the model
    vec = model[word]

    # find and return poems that are most similar to that
    # word vector
    similar_poems = model.docvecs.most_similar([vec],
                                               topn=n)

    return similar_poems


# search based on a phrase
def phrase_similarity(phrase, df, model, n=5):

    '''
    Function to find the n-most-similar poems, based on
    a document vector created by the input model.


    Input
    -----
    text : str
        Words to use to create a document vector and
        compare to poem document vectors.

    df : Pandas DataFrame
        Database of all poems.

    model : Doc2Vec model
        Fitted Gensim Doc2Vec object.
            `gensim.models.doc2vec.Doc2Vec`


    Optional input
    --------------
    n : int
        The number of poems to return (default=5).


    Output
    ------
    similar_poems : list (tup)
        List of similar poems with poem index as an integer
        and percent similarity as a float.

    '''

    # process the input in the same manner of documents in
    # the model
    words = simple_process(phrase).split()

    # create a vector for the input text based on the model
    vec = model.infer_vector(words)

    # find and return poems that are most similar to that
    # word vector
    similar_poems = model.docvecs.most_similar([vec],
                                               topn=n)

    return similar_poems


# search based on a poem in the dataframe
def poem_similarity(
    title,
    poet,
    df_info,
    df_vectors,
    model,
    n=5
):

    '''
    Function to find the n-most-similar poems, based on
    cosine similarity scores.


    Input
    -----
    title : str
        Title of input poem, for which to find the most
        similar poems.

    poet : str
        Author of poem.

    df_info : Pandas DataFrame
        Database with poet, title, URL, and genre.

    df_vectors : Pandas DataFrame
        Database of poem data and embeddings
            (Doc2Vec or Word2Vec).


    Optional input
    --------------
    n : int
        The number of poems to return (default=5).


    Output
    ------
    similar_poems : list (tup)
        List of similar poems with poem index as an integer
        and percent similarity as a float.

    '''

    # find the index value for the input poem
    # NOTE: since some poems have the same title but
    #       different poets, both fields are required
    poem_id = df_info[(df_info.title == title) &
                      (df_info.poet == poet)].index[0]

    # calculate cosine similarities for that poem
    # NOTE: index value should correspond to same poem in
    #       both dataframes
    cos_sims = enumerate(cosine_similarity(
        df_vectors.iloc[poem_id].values.reshape(1, -1),
        df_vectors)[0]
                        )

    # find and return poems that are most similar to the
    # input poem
    # NOTE: add one to the `n` value and slice off first
    #       result because the first result will always be
    #       the same as the input poem
    similar_poems = sorted(cos_sims,
                           key=itemgetter(1),
                           reverse=True)[1:n+1]

    return similar_poems


# filter recommended poems based on various parameters
def poem_filter(
    similar_poems,
    df,
    genre=None,
    min_lines=None,
    max_lines=None,
    min_len_line=None,
    max_len_line=None,
    polarity=None,
    end_rhyme=None
):

    '''
    Function to filter results based on various optional
    parameters.


    Input
    -----
    similar_poems : list (tup)
        List of document tags (corresponding to dataframe index
        values) and percentage of cosine similarity.

    df : Pandas DataFrame
        Database of poems and info.


    Optional input
    --------------
    genre : str
        Genre of returned poems.
        One of ['beat', 'black_arts_movement', 'black_mountain',
                'confessional', 'harlem_renaissance', 'imagist',
                'language_poetry', 'modern', 'new_york_school',
                'new_york_school_2nd_generation', 'objectivist',
                'romantic', 'victorian'].

    min_lines : int
        Minimum number of lines in returned poem.

    max_lines : int
        Maximum number of lines in returned poem.

    min_len_line : float
        Minimum average number of words per line in returned
        poem.

    max_len_line : float
        Maximum average number of words per line in returned
        poem.

    polarity : str
        Sentiment of poem.
        One of ['positive', 'neutral', 'negative'].

    end_rhyme : str
        Whether returned poems have few to no end rhymes (`no`)
        or many end rhymes (`yes`).
        One of ['no', 'yes'].


    Output
    ------
    similar_poems : list (tup)
        Filtered list of tuples with poem index as an integer
        and percent similarity as a float.

    Prints a message if similar_poems is empty.

    '''

    # genre filter
    if genre:
        # limit dataframe to poems within input genre
        df = df[df.genre == genre]

    # poem length filter
    if min_lines:
        if max_lines:
            # if user inputs both values
            df = df[(df.num_lines >= min_lines) &
                    (df.num_lines <= max_lines)]

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
            df = df[(df.avg_len_line >= min_len_line) &
                    (df.avg_len_line <= max_len_line)]
        else:
            # if user only inputs minimum length of line
            df = df[df.avg_len_line >= min_len_line]

    # if user only inputs minimum length of line
    elif max_len_line:
        df = df[df.avg_len_line <= max_len_line]

    # sentiment filter
    if polarity:
        # limit dataframe to poems within input polarity
        df = df[df.sentiment_polarity == polarity]

    # end rhyme filter
    # limit dataframe to poems within that end_rhyme value
    if end_rhyme is False:
        df = df[df.end_rhyme == 0]
    elif end_rhyme is True:
        df = df[df.end_rhyme == 1]

    # re-create the original list using only poems that satisfy
    # the filters (i.e. appear in the filtered dataframe)
    similar_poems = [(i, pct) for i, pct in similar_poems if i in df.index]

    # return poems if available
    if similar_poems:
        return similar_poems

    # return a message if the list is empty
    else:
        print('Filter too fine. Please retry.')             


# filter recommended poems based on various parameters
def filter_process(similar_poems, df):

    '''
    Function to run filter on Streamlit page.


    Input
    -----
    similar_poems : list (tup)
        List of document tags (corresponding to dataframe index
        values) and percentage of cosine similarity.

    df : Pandas DataFrame
        Database of poems and info.

    '''

    # a fateful decision, to filter or not to filter
    to_filter_option = st.radio(
        'Would you like to use my filter?',
        ['no', 'yes'])

    # filter parameters in the sidebar
    if to_filter_option == 'yes':
        # title
        st.sidebar.markdown('## FILTER OPTIONS')

        # instantiate a blank option and add all genres from
        # dataframe as other options
        genres = ['']
        genres.extend(df['genre'].unique())
        # reformat genre titles
        genres = [genre.replace('_', ' ').title() for genre in genres]

        # selected genre (if selected), formatted back to be
        # used in dataframe
        genre_option = st.sidebar.selectbox(
            'Pick a genre:', genres).replace(' ', '_').lower()

        # range slider with minimum and maximum poem length
        # values
        num_lines_options = st.sidebar.slider(
            'Select a range for the number of lines in each \
            recommendation.',
            # use dataframe's min and max values as ends to slider
            int(df.num_lines.min()), int(df.num_lines.max()),
            (int(df.num_lines.min()), int(df.num_lines.max())))

        # range slider with minimum and maximum line length values
        len_line_options = st.sidebar.slider(
            "Select a range for the average number of words per \
            line in each recommendation.",
            # use dataframe's min and max values as ends to slider
            df.avg_len_line.min(), df.avg_len_line.max(),
            (df.avg_len_line.min(), df.avg_len_line.max()))

        # instantiate a blank option and add polarities from
        # dataframe as other options
        polarities = ['']
        polarities.extend(df['sentiment_polarity'].unique())

        # selected sentiment (if selected)
        polarity_option = st.sidebar.selectbox(
            'Pick a sentiment:', polarities)

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
            poem_printout(df, similar_poems)
        # if no poems pass through the filter, message the user
        else:
            st.markdown('#### Filter too fine. Please retry.')

    # if no filter desired, print out the recommendations
    else:
        poem_printout(df, similar_poems)
