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

# custom functions for this app
from functions_app import *


# load vector model
model = Doc2Vec.load('data/doc2vec.model')

# load dataframe
with gzip.open('data/poetry_rec_system.pkl', 'rb') as hello:
    df = pickle.load(hello)


# image of PO-REC
st.image('data/PO-REC.png', width=300,
		 # use_column_width=True
		 ) 

# message from the recommender-bot
st.title('Greetings! It is I, PO-REC.')
st.header('I am designed to recommend poetry based on certain parameters.')
st.subheader('You can fiddle with my settings on the left of your screen.')


# number of poem recommendations in sidebar
# NOTE: text in separate markdown because couldn't figure out
# how to change font size within number_input
st.sidebar.markdown('#### How many poems shall I compute?')
num_option = st.sidebar.number_input(
	'',
	# set min/max values and default value of 100
	min_value=1, max_value=len(df), value=100)


# format blank space
st.sidebar.markdown('')


# select a function to run, word_similarity, text_similarity, or poem_similarity
st.sidebar.markdown('#### What method shall I use to compute?')
initialize_option = st.sidebar.radio(
	'',
	['word', 'text', 'poem'])


# format blank space
st.sidebar.markdown('')


# for word option
if initialize_option == 'word':

	# ask user for a word
	word_option = st.text_input(
	'Give me one word.')

	# upon user input
	if word_option:

		# determine if word (reformatted) in model's vocabulary
		if word_option.lower() in model.wv.vocab.keys():

			# run function
			similar_poems = word_similarity(word_option.lower(), df, model,
											num_poems=num_option)

			# filter
			filter_process(similar_poems, df)

		# PO-REC's message if word not in model's vocabulary
		else:
			st.markdown(f'### It may surprise you to learn that I do not know the word\
				***{word_option}***.')
			st.markdown(f'### Please try another.')


# for text option
elif initialize_option == 'text':

	# ask user for words
	text_option = st.text_input(
	'Give me some words.')

	# upon user input
	if text_option:

		# run function
		similar_poems = text_similarity(text_option, df, model, num_poems=num_option)
		
		# filter
		filter_process(similar_poems, df)


# for poem option
elif initialize_option == 'poem':

	# initialize blank list
	poets = ['']
	# add all poets from dataframe
	poets.extend(df['poet'].unique())
	# prompt user to select poet
	poet_option = st.selectbox(
	    'Pick a poet:',
	     poets)

	# initialize blank list
	poet_titles = ['']
	# add all titles from that poet
	poet_titles.extend(df[df.poet == poet_option].title.unique())
	# prompt user to select title (only after poet is selected)
	if poet_option:
		title_option = st.selectbox(
		    'Pick a poem:',
		     poet_titles)

		# upon title selection
		if title_option:

			# run function
			similar_poems = poem_similarity(title_option,
											poet_option,
											df, model,
											num_poems=num_option)

			# filter
			filter_process(similar_poems, df)
