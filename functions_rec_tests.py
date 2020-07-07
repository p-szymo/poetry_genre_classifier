import numpy as np
import pandas as pd

# print recommended poems
def poem_printout(similar_poems):
    
    '''Input a list of tuples of document tags (corresponding to dataframe index values) and percentage of similarity.
       Output a formatted list of poem titles with corresponding poet, link, and percent match.'''
    
    # separation line
    print('-'*100)
    
    # loop over list of tuples
    for i, pct in similar_poems:
        # similarity of match as a percent.
        print(f'{round(pct*100,1)}% match')
        # title of poem and poet name
        print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
        # genre of poem
        print(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
        # corresponding URL to PoetryFoundation.org page
        print(f'URL: {df.loc[i,"poem_url"]}')
        # separation line
        print('-'*100)

              
# search based on a keyword
def word_similarity(word, df, model, num_poems=5, to_print=True):
    
    '''Input a single word, as well as dataframe, Doc2Vec model, number of poems to recommend (default is 5),
       and whether or not to include printout.
       Output a list of tuples with document tag and percent match, to be fed into poem_printout function.'''
    
    # find the vector for that word, if it exists within the model
    vec = model[word]
    
    # find poems that are most similar to that word vector
    similar_poems = model.docvecs.most_similar([vec], topn=num_poems)
    
    # printout (if desired)
    if to_print:
        poem_printout(similar_poems)
    
    # return recommended poems
    return similar_poems   
             
              
# search based on a phrase              
def text_similarity(text, df, model, num_poems=5, to_print=True):
                  
    '''Input a text of any length, as well as dataframe, Doc2Vec model, and number of poems to recommend (default is 5),
       and whether or not to include printout.
       Output a list of tuples with document tag and percent match, to be fed into poem_printout function.'''
              
    # process the input in the same manner of documents in the model
    words = simple_preprocess(text)
                  
    # create a vector for the input text based on the model
    vec = model.infer_vector(words)
                  
    # find poems that are most similar to that vector
    similar_poems = model.docvecs.most_similar([vec], topn=num_poems)
    
    # printout (if desired)
    if to_print:
        poem_printout(similar_poems)
    
    # return recommended poems
    return similar_poems         
              
              
# search based on a poem in the dataframe              
def poem_similarity(title, df, model, num_poems=5, to_print=True):
              
    '''Input a title of a poem and its corresponding poet, as well as dataframe, Doc2Vec model, number 
       of poems to recommend (default is 5), and whether or not to include printout.
       Output a list of tuples with document tag and percent match, to be fed into poem_printout function.'''
    
    # find the index value for the poem
    poem_id = df[df['title'] == title].index[0]
    
    # find the vector for that poem
    # NOTE: index value and document tag should be the same
    vec = model.docvecs[poem_id]
              
    # find and return poems that are most similar to that word vector
    # add one to the input number of poems and slice off first result because the first result will always be
    # the same as the input poem
    similar_poems = model.docvecs.most_similar([vector], topn=num_poems+1)[1:]
              
    # printout (if desired)
    if to_print:
        poem_printout(similar_poems)
    
    # return recommended poems
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
       end_rhyme: one of ['no', 'yes'], 'no' being poems with few to no end rhymes and 'yes' being a poem with end rhymes
       
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
    # NOTE: input is 'no' or 'yes' for user readability
    # convert to 0 or 1 and limite dataframe to poems within that end_rhyme value
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
               
              
# create genre subsets for T-SNE visualizations                  
def make_tsne_subset(tsne_df, poetry_df, column, col_value):
              
    '''Modeled after https://github.com/aabrahamson3/beer30/blob/master/functions.py
       Input a dataframe of a fit/transformed t-SNE object, with document tags as the index,
           a dataframe of poems, a column name, and a value for that column.
       Output a dataframe with t-SNE coordinates of that specific column value, to show clustering.'''
    
    # limit dataframe to column with column value
    subset = poetry_df.loc[poetry_df[column] == col_value]
              
    # create a set of subsets indices
    subset_set = set(subset.index)
        
    # match those indices with corresponding indices in T-SNE dataframe
    match = set(tsne_df.index).intersection(subset_set)
    
    # create a return tsne_df with corresonding indicies
    style_subset = tsne_df[tsne_df.index.isin(match)]
    return style_subset