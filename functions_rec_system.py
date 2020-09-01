import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from functions import simple_process


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
    
    # separation line
    print('-'*100)
    
    # loop over list of tuples
    for i, pct in similar_poems:
        
        # similarity of match as a percent.
        print(f'{round(pct*100,1)}% match')
        
        # title of poem and poet name
        print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
        
        # genre of poem
        if df.loc[i,"genre"] != 'new_york_school_2nd_generation':
            print(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
        # special case
        else:
            print('GENRE: New York School 2nd Generation')
        # corresponding URL to PoetryFoundation.org page
        print(f'URL: {df.loc[i,"poem_url"]}')
        # separation line
        print('-'*100)

              
# search based on a keyword
def word_similarity(
    word, 
    df, 
    model, 
    n=5, 
    to_print=True):
    
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
        
    to_print : bool
        Whether to print poem similarities in stylized
        format (default=True).
    
    
    Output
    ------
    similar_poems : list (tup)
        List of similar poems with poem index as an integer
        and percent similarity as a float.
        
    '''
    
    # if word in model's corpus
    try:
        # find vector for input word, if it exists within the model
        vec = model[word]

        # find poems that are most similar to that word vector
        similar_poems = model.docvecs.most_similar([vec], topn=n)

        # optional printout
        if to_print:
            poem_printout(df, similar_poems)

        return similar_poems
    
    # if word not in model's corpus
    except KeyError:
        print("I don't know that word; try again.")
             
              
# search based on a phrase              
def phrase_similarity(
    text, 
    df, 
    model, 
    n=5, 
    to_print=True):
                  
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
        
    to_print : bool
        Whether to print poem similarities in stylized
        format (default=True).
    
    
    Output
    ------
    similar_poems : list (tup)
        List of similar poems with poem index as an integer
        and percent similarity as a float.
        
    '''
              
    # process the input in the same manner of documents in 
    # the model
    words = simple_process(text).split()
                  
    # create a vector for the input text based on the model
    vec = model.infer_vector(words)
                  
    # find poems that are most similar to that vector
    similar_poems = model.docvecs.most_similar([vec], topn=n)
    
    # optional printout
    if to_print:
        poem_printout(df, similar_poems)
    
    return similar_poems
              
              
# search based on a poem in the dataframe              
def poem_similarity(
    title, 
    poet, 
    df_info,
    df_vectors,
    n=5, 
    to_print=True):
                    
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
        Database of poet, title, URL, and genre.
        
    df_vectors : Pandas DataFrame
        Database of poem data and embeddings
            (Doc2Vec or Word2Vec).
        
        
    Optional input
    --------------
    n : int
        The number of poems to return (default=5).
        
    to_print : bool
        Whether to print poem similarities in stylized
        format (default=True).
    
    
    Output
    ------
    similar_poems : list (tup)
        List of similar poems with poem index as an integer
        and percent similarity as a float.
        
    '''
                    
    # find the index value for the input poem
    # NOTE: since some poems have the same title but 
    #       different poets, both fields are required
    poem_id = df_info[(df_info.title == title) & \
                      (df_info.poet == poet)].index[0]
    
    # calculate cosine similarities for that poem
    # NOTE: index value should correspond to same poem in
    #       both dataframes
    cos_sims = enumerate(cosine_similarity(
        df_vectors.iloc[poem_id].values.reshape(1,-1), 
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
              
    # optional printout
    if to_print:
        poem_printout(df_info, similar_poems)
              
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
    end_rhyme=None,
    to_print=True):
                    
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
            df = df[(df.num_lines >= min_lines) & \
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
            df = df[(df.avg_len_line >= min_len_line) & \
                    (df.avg_len_line <= max_len_line)]
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
    # NOTE: input is 'no' or 'yes' for user readability
    # convert to 0 or 1 and limite dataframe to poems within that end_rhyme value
    if end_rhyme == False:
        df = df[df.end_rhyme == 0]
    elif end_rhyme == True:
        df = df[df.end_rhyme == 1]
    
    # re-create the original list using only poems that satisfy the filters (i.e. appear in the filtered dataframe)
    similar_poems = [(i, pct) for i, pct in similar_poems \
                         if i in df.index]
    
    # return poems if available
    if similar_poems:
              
        # optional printout
        if to_print:
            poem_printout(df, similar_poems)
              
        return similar_poems
                    
    # return a message if the list is empty
    else:
        print('Filter too fine. Please retry.')
               
              
# create genre subsets for t-SNE visualizations                  
def make_tsne_subset(tsne_df, poetry_df, column, col_value):
              
    '''
    Function to create subsets to prepare for t-SNE
    visualization.
    
    Input
    -----
    tsne_df : Pandas DataFrame
        Fit/transformed t-SNE object, with document tags
        as the index.
    
    poetry_df : Pandas DataFrame
        Database of poems.
    
    column : str
        Name of column on which to create subset.
        
    col_value : str
        Value of column on which to create subset.
      
      
    Output
    ------
    style_subset : Pandas DataFrame
        DataFrame subset.
        
        
    
        
    [Modeled after]: 
    https://github.com/aabrahamson3/beer30/blob/master/functions.py
    
    '''
    
    # limit dataframe to column with column value
    subset = poetry_df.loc[poetry_df[column] == col_value]
              
    # create a set of subsets indices
    subset_set = set(subset.index)
        
    # match those indices with corresponding indices in 
    # TtSNE dataframe
    match = set(tsne_df.index).intersection(subset_set)
    
    # create and return tsne_df with corresonding indices
    style_subset = tsne_df[tsne_df.index.isin(match)]
              
    return style_subset
              

# search based on a poem in the dataframe   
# if similarities have been calculated beforehand
# NOTE: faster but requires a large file (114MB)
def poem_similarity_precalculated(
    title, 
    poet, 
    df, 
    similarities, 
    n=5, 
    to_print=True):
                    
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
        
    df : Pandas DataFrame
        Database of all poems.
        
    similarities : list (arr)
        List of arrays with cosine similarity scores.
        
        
    Optional input
    --------------
    n : int
        The number of poems to return (default=5).
        
    to_print : bool
        Whether to print poem similarities in stylized
        format (default=True).
    
    
    Output
    ------
    similar_poems : list (tup)
        List of similar poems with poem index as an integer
        and percent similarity as a float.
        
    '''
                    
    # find the index value for the input poem
    # NOTE: since some poems have the same title but 
    # different poets, both fields are required
    poem_id = df[(df.title == title) & (df.poet == poet)].\
                  index[0]
    
    # find the list of cosine similarities for that poem
    # NOTE: index value and document tag should be the same
    cos_sims = enumerate(similarities[poem_id])
    
    # find and return poems that are most similar to the 
    # input poem
    # add one to the input number of poems and slice off 
    # first result because the first result will always be
    # the same as the input poem
    similar_poems = sorted(cos_sims, 
                           key=itemgetter(1), 
                           reverse=True)[1:n+1]
              
    # optional printout
    if to_print:
        poem_printout(df, similar_poems)
              
    return similar_poems