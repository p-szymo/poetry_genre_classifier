import pandas as pd
import string


def word_similarity(word, df, model, num_poems=5, to_print=True):
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
              
def text_similarity(text, df, model, num_poems=5, to_print=True):
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
              
def poem_similarity(title, df, model, num_poems=5, to_print=True):
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
                  
def poem_filter(similar_poems, df, genre=None, min_lines=None, max_lines=None, min_len_line=None, max_len_line=None,
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
                  
                  
def make_tsne_subset(tsne_df: pd.DataFrame, poetry_df: pd.DataFrame, column: str, genre: str) -> pd.DataFrame:
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