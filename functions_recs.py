def word_similarity(word, num_poems=5):
    # search based on a keyword
    vec = model[word]
    similar_poems = model.docvecs.most_similar([vec], topn=num_poems)
    for i, pct in similar_poems:
        print(f'{round(pct*100,1)}% match')
        print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
        print(f'URL: {df.loc[i,"poem_url"]}')
        print('-'*100)
              
              
def text_similarity(text, num_poems=5):
    # search based on a string
    words = text.translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits))\
                     .strip().lower().split()
    vector = model.infer_vector(words)
    similar_poems = model.docvecs.most_similar([vector], topn=num_poems)
    for i, pct in similar_poems:
        print(f'{round(pct*100,1)}% match')
        print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
        print(f'URL: {df.loc[i,"poem_url"]}')
        print('-'*100)
              
              
def poem_similarity(title, num_poems=5):
    poem_id = df[df['title'] == 'On Cowee Ridge'].index[0]
    poem = df.loc[poem_id, 'string_titled']
    poem_words = poem.translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits))\
                     .strip().lower().split()
    vector = model.infer_vector(poem_words)
    similar_poems = model.docvecs.most_similar([vector], topn=num_poems)
    for i, pct in similar_poems:
        print(f'{round(pct*100,1)}% match')
        print(f'{df.loc[i,"title"].upper()} by {df.loc[i,"poet"]}')
        print(f'URL: {df.loc[i,"poem_url"]}')
        print('-'*100)