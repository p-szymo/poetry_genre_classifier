import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import gensim
from gensim.models import Doc2Vec

import pickle
import gzip

from functions_recs import *

model = Doc2Vec.load('data/doc2vec.model')

# uncomment to load
with gzip.open('data/poetry_rec_system.pkl', 'rb') as hello:
    df = pickle.load(hello)

st.title('Greetings! It is I, PO-REC.')
st.text('Pick an option or two.')

# option = st.selectbox(
#     'Which poet do you like best?',
#      df['poet'].unique())

# 'Here are available poem titles.'
# [title for title in df[df.poet == option].title]

poets = ['--']
poets.extend(df['poet'].unique())
option_poet = st.sidebar.selectbox(
    'Pick a poet:',
     poets)

poet_titles = ['--']
all_titles = ['--']
poet_titles.extend(df[df.poet == option_poet].title.unique())
all_titles.extend(df['title'].unique())
if option_poet == '--':
	option_title = st.sidebar.selectbox(
	    'Pick a poem:',
	     all_titles)

else:
	option_title = st.sidebar.selectbox(
	    'Pick a poem:',
	     poet_titles)
	
st.text(df.loc[df.title == option_title, 'poem_url'])
# 'Here are available poem titles.'
# [title for title in df[df.poet == option].title]


def word_similarity(word, df, model, num_poems=5, to_print=True):
    # search based on a keyword
    vec = model[word]
    similar_poems = model.docvecs.most_similar([vec], topn=num_poems)
    if to_print:
        for i, pct in similar_poems:
            st.markdown(f'### {round(pct*100,1)}% match')
            st.markdown(f'[{df.loc[i,"title"].upper()}]({df.loc[i,"poem_url"]}) by {df.loc[i,"poet"]}')
            st.markdown(f'GENRE: {df.loc[i,"genre"].replace("_", " ").title()}')
            st.markdown('-'*75)
    return similar_poems   

similar_poems = word_similarity('hell', df, model)