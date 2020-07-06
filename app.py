import streamlit as st
import numpy as np
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess

import pickle
import gzip

from functions_recs import *

model = Doc2Vec.load('data/doc2vec.model')

# uncomment to load
with gzip.open('data/poetry_rec_system.pkl', 'rb') as hello:
    df = pickle.load(hello)


st.title('Greetings! It is I, PO-REC.')
st.markdown('### I am designed to recommend poetry based on certain parameters.')

# option = st.selectbox(
#     'Which poet do you like best?',
#      df['poet'].unique())

# 'Here are available poem titles.'
# [title for title in df[df.poet == option].title]

# st.button('word')

num_option = st.sidebar.number_input(
	'How many poems shall I compute?',
	min_value=3, max_value=len(df))

initialize_option = st.sidebar.radio(
	'What method shall I use to compute?',
	['word', 'text', 'poem'])

# initialize_option = st.selectbox(
# 	'How may I compute?',
# 	['--', 'word', 'text', 'poem title'])

if initialize_option == 'word':

	word_option = st.text_input(
	'Give me one word.')

	if word_option:

		if word_option.lower() in model.wv.vocab.keys():

			similar_poems = word_similarity(word_option.lower(), df, model,
											num_poems=num_option)

			filter_process(similar_poems, df)

		else:
			st.markdown(f'### It may surprise you to learn that I do not know the word,\
				{word_option}.')
			st.markdown(f'### Please try another.')


elif initialize_option == 'text':

	text_option = st.text_input(
	'Give me some words.')

	if text_option:

		similar_poems = text_similarity(text_option, df, model, num_poems=num_option)
		
		filter_process(similar_poems, df)


elif initialize_option == 'poem':

	poets = ['']
	poets.extend(df['poet'].unique())
	poet_option = st.selectbox(
	    'Pick a poet:',
	     poets)

	poet_titles = ['']
	poet_titles.extend(df[df.poet == poet_option].title.unique())
	if poet_option:
		title_option = st.selectbox(
		    'Pick a poem:',
		     poet_titles)

		if title_option:

			

			similar_poems = poem_similarity(title_option,
											poet_option,
											df, model,
											num_poems=num_option)

			filter_process(similar_poems, df)


	
# st.text(df.loc[df.title == option_title, 'poem_url'])
# 'Here are available poem titles.'
# [title for title in df[df.poet == option].title]


 
