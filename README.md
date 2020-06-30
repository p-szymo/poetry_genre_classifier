# Predicting Poetic Movements

## Approach
After scraping [Poetry Foundation](https://www.poetryfoundation.org/) for poems within certain genres, I perform natural language processing (NLP) techniques to analyze the characteristics and word usage of four self-determined poetic movements: pre-1900 (Victorian and Romantic), Modern (a standalone category), Metropolitan (New York School [1st and 2nd Generation], Confessional, Beat, Harlem Renaissance, Black Arts Movement), and Avant-Garde (Imagist, Black Mountain, Language Poetry, Objectivist). Through text processing, feature engineering, and exploratory data analysis, I discover insights into how important words and structure relate to genre. I then create predictive models to provide further insight and confirm my findings during EDA.

### Some questions:
* How does the sentiment of tweets change over time?
    * Hypothesis: Tweets will be more negative on average in January and get more positive on average as time goes on.
* Will Twitter stats (number of likes, replies, retweets) play a role in determining sentiment?
    * Hypothesis: The most important features will most likely be the words themselves.
* Does topic modeling provide any insight toward tweet sentiment or the COVID-19 crisis?
    * Hypothesis: Topic modeling should be a factor in determining sentiment and can give us insights into the pandemic.
* What insights can be provided by using machine learning?
    * Hypothesis: The lion's share of the insights will come during EDA.
* What are the most frequent words? And do they play a role in determining sentiment?

## Findings and future considerations
- Pre-1900 poetry is easily recognized
    - High number of end rhymes, most words per line, and simpler words (fewer syllables per word)
- Avant-garde is the polar opposite
    - Practically no end rhymes, fewest words per line, and more complex words (more syllables per word)
- Poetry is rarely neutral (and generally positive) and fairly equally subjective/objective
- Lots of visual and temporal vocabulary
- Form/structure is important for prediction (especially Random Forest models)
    - Further exploration desired
        - Other types of rhyme
        - Use of line breaks, tabs, and spacing
        - Topic modeling
- SVM relies mostly on vocabulary
    - Further exploration desired
        - Word embeddings (self-trained and pre-trained)
        - POS tagging

- Some of the words that the SVM model weighed the heaviest were surprising, given that they were mostly not among the top words overall:
    - among
    - black
    - get
    - dream
    - look
    - cry
    - crazy
    - leaf
    - sing

## 10 most common words (after removing stopwords):
    'love'  (2996)
    'say'   (2528)
    'day'   (2345)
    'see'   (2275) 'make': 2260, 'eye': 2091, 'know': 2013, 'night': 1977, 'life': 1975, 'man': 1911
## Most prevalent features in the Random Forest model: (in order)
### 10 most common words (after removing stopwords):
    'need'
    'spread'
    'protect'
    'make'
    'help'
    'say'
    'glove'
    'public'
    'hospital'
    'new'

### 10 best features (Decision Tree Classifier):
    Subjectivity score  (0.0611)
    Number of likes     (0.0139)
    'protect'           (0.0132)
    'help'              (0.0129)
    'infected'          (0.0115)
    'safe'              (0.0094)
    'please'            (0.0083)
    'death'             (0.0083)
    'hand'              (0.0076)
    Number of replies   (0.0072)

# Final conclusion
##### The overall sentiment of tweets was fairly evenly divided between positive and negative throughout the five months. There were some interesting results from our prediction models, namely that some continuous variables like subjectivity score, number of likes, and number of replies were some of the most important variables for predicting a tweet's sentiment. Other important features were words with high frequencies. Given more time we would try to get better accuracy via a deep learning model, including an LSTM model. And finally, we would like to further investigate sentiment toward the work mask (or masks) in particular as opposed to the overall sentiment of the tweet as a whole.

## List of files
- **.gitignore** - list of files and pathways to ignore
- **data_cleaning_notebook.ipynb** - notebook of compiling our dataframes
- **eda_visualizations_notebook.ipynb** - notebook with EDA and chart/visualization creations
- **functions.py** - file with functions used in this project
- **modeling_notebook.ipynb** - notebook with Naive Bayes and Decision Tree models
- **nlp_features_notebook.ipynb** - notebook with text processing, LDA topic modeling, and subjectivity scoring
- **presentation.pdf** - slides for our presentation of this project
- **README.md** - this very file!
- **twitter_scraping_notebook.ipynb** - notebook detailing our scraping of tweets
- **archives** folder - old jupyter notebooks, mostly scrap
- **images** folder - charts and visualizations created during the project
- **models** folder - tweet count vectors and a couple of lda topic models (note: unfortunately not the one in our presentation)

## Visualizations
- Decision Tree Confusion Matrix:
![Decision Tree Confusion Matrix](Images/dt_conf_matrix.png)

- Tweet Sentiment by LDA Topic:
![Tweet Sentiment by LDA Topic](Images/lda_sentiment_stacked_bar.png)

- LDA Topics:
![LDA Topics](Images/lda_topics.png)

- Sentiment Distribution of Top 20 Tweets per Day:
![Sentiment Distribution of Top 20 Tweets per Day](Images/sentiment_stacked_line_top20.png)

- Sentiment Distribution Over Time:
![Sentiment Distribution Over Time](Images/sentiment_stacked_line.png)

- Top 25 Words by Frequency:
![Top 25 Words by Frequency](Images/top25_words_bar_twitter-blue.png)

- Topic Distribution Over Time:
![Topic Distribution Over Time](Images/topic_distribution_over_time.png)

- Number of Negative Tweets by Topic:
![Number of Negative Tweets by Topic](Images/tweet_count_by_topic_neg.png)

- Number of Neutral Tweets by Topic:
![Number of Neutral Tweets by Topic](Images/tweet_count_by_topic_neu.png)

- Number of Positive Tweets by Topic:
![Number of Positive Tweets by Topic](Images/tweet_count_by_topic_pos.png)

- Word Cloud of Top 100 Words:
![Word Cloud of Top 100 Words](Images/wordcloud_top100.jpg)

- Word Cloud:
![Word Cloud](Images/wordcloud.jpg)



### BLOG POST FORTHCOMING

