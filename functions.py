# dataframe packages
import pandas as pd
import numpy as np
import itertools

# visualization package
import matplotlib.pyplot as plt

# string processing
import re
import string
from ast import literal_eval

# nlp packages
import nltk
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from textblob import TextBlob as tb
import pronouncing

# statistical packages
from scipy.stats import ttest_ind


# convert lists that became strings when saved to csv
def destringify(x):
    
    '''
    Function using AST's `literal_eval` to convert a list inside 
    of a string into a list.
    
    Allows for errors, namely those caused by NaN values.
    
    Input
    -----
    x : str
        String with a list inside.
        
    Output
    ------
    x : list
        The list rescued from within a string.
        Returns the input object if an error occurs.
    
    [Code found on Stack Overflow]:
    https://stackoverflow.com/questions/52232742/how-to-use-ast-\
    literal-eval-in-a-pandas-dataframe-and-handle-exceptions
    '''
    try:
        return literal_eval(x)
    except (ValueError, SyntaxError) as e:
        return x
    
# list of roman numerals for stop words
def roman_numerals():
    
    '''
    Returns a list of Roman numerals, 1-100.
    '''
    
    return [
            'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 
            'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 
            'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 
            'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI', 
            'XXXII', 'XXXIII', 'XXXIV', 'XXXV', 'XXXVI', 'XXXVII', 
            'XXXVIII', 'XXXIX', 'XL', 'XLI', 'XLII', 'XLIII', 'XLIV', 
            'XLV', 'XLVI', 'XLVII', 'XLVIII', 'XLIX', 'L', 'LI', 
            'LII', 'LIII', 'LIV', 'LV', 'LVI', 'LVII', 'LVIII', 
            'LIX', 'LX', 'LXI', 'LXII', 'LXIII', 'LXIV', 'LXV', 
            'LXVI', 'LXVII', 'LXVIII', 'LXIX', 'LXX', 'LXXI', 
            'LXXII', 'LXXIII', 'LXXIV', 'LXXV', 'LXXVI', 'LXXVII', 
            'LXXVIII', 'LXXIX', 'LXXX', 'LXXXI', 'LXXXII', 'LXXXIII', 
            'LXXXIV', 'LXXXV', 'LXXXVI', 'LXXXVII', 'LXXXVIII', 
            'LXXXIX', 'XC', 'XCI', 'XCII', 'XCIII', 'XCIV', 'XCV', 
            'XCVI', 'XCVII', 'XCVIII', 'XCIX', 'C'
        ]


# clean up poem formatting
def line_cleaner(lines):
    
    '''
    Function that removes whitespace, section headers, and empty
    lines from list of strings.
    
    Input
    -----
    lines : list (str)
        List of strings to be cleaned.
        
    Output
    ------
    lines_clean : list (str)
        List of strings without whitespace, section headers, or
        empty strings.
    '''
    
    # remove spaces at beginning and end of lines
    lines_clean = [line.strip() for line in lines]
    
    # create a list of section headers to remove
    section_headers = roman_numerals()
    section_headers += [str(i) for i in range(1,101)]
    section_headers += [f'[{i}]' for i in range(1,101)]
    section_headers += [f'[{i}]' for i in range(1,101)]
    section_headers += [f'PART {num}' for num in roman_numerals()]
    section_headers += [f'PART {i}' for i in range(1,101)]
    
    # remove section headers
    lines_clean = [line for line in lines_clean\
                   if line not in section_headers]
    
    # remove any empty strings
    lines_clean = [line for line in lines_clean if line]
    
    return lines_clean


# count number of words in a text
def word_counter(lines):
    
    '''
    Function to count the number of words in a list of strings.
    
    Input
    -----
    lines : list (str)
        List of strings to count.
    
    Output
    ------
    word_count : int
        Total number of words across all strings.
    '''
    
    # calculate the number of words per line
    line_count = [len(line.split()) for line in lines]
    
    # add up the word counts of each line
    word_count = sum(line_count)
    
    return word_count


# count number of end rhymes in a text
def end_rhyme_counter(lines):
    
    '''
    Function to count the instances of rhymes that occur among
    the last words in lines (end rhymes).
    
    Input
    -----
    lines : list (str)
        List of strings to compare.
    
    Output
    ------
    sum(rhyme_counts) : int
        The number of end rhymes.
    '''
    
    # instantiate an empty dictionary
    rhyme_dict = {}
    
    # make a list of words at the end of the line
    end_words = [line.split()[-1].translate\
                 (str.maketrans('', '', string.punctuation)) \
                 for line in lines]
    
    # loop to build the dictionary
    for word in end_words:
        for i in range(len(end_words)):
            
            # check if a word rhymes with another word in the list
            if end_words[i] in pronouncing.rhymes(word):
                
                # check if word is already a key in the dictionary
                if word not in rhyme_dict:
                    
                    # or if its rhyming word is already a key 
                    # in the dictionary
                    if end_words[i] not in rhyme_dict:
                        
                        # if neither is, create the word as key and 
                        # it's rhyme as a value (in a list)
                        rhyme_dict[word] = [end_words[i]]
                        
                else:
                    # if word is already a key, append its rhyme to 
                    # its value
                    rhyme_dict[word].append(end_words[i])
                    
    # count up the amount of (unique) rhymes per word
    rhyme_counts = [len(rhyme) for rhyme in rhyme_dict.values()]
    
    return sum(rhyme_counts)


# count total syllables in text
def syllable_counter(lines):
    
    '''
    Function to count all syllables in a list of strings.
    
    NOTE: This does not factor in multi-syllabic digits, 
    times (i.e. 1:03), and most likely other non-"word" words.
    
    Input
    -----
    lines : list (str)
        List of strings to count.
    
    Output
    ------
    sum(total) : int
        Total number of syllables in the input list.
       
    [Modified Allison Parrish example in documention for her 
     library, pronouncing]:
    https://pronouncing.readthedocs.io/en/latest/tutorial.html\
    #counting-syllables
    '''
    # create empty list
    total = []
    
    # loop over list
    for line in lines:
        
        # turn each word into a string of its phonemes
        # if else statement ensures that each word is counted with 
        # at least one syllable, even if that word is not in the 
        # pronouncing library's dictionary (using phoneme for 'I' 
        # as a placeholder for single syllable)
        phonemes = [pronouncing.phones_for_word(word)[0] \
                    if pronouncing.phones_for_word(word) \
                    else 'AY1' for word in line.split()]
        # count the syllables in each string and add the total 
        # syllables per line to the total list
        total.append(sum([pronouncing.syllable_count(phoneme)\
                          for phoneme in phonemes]))
    
    # return the total number of syllables
    return sum(total)


# contractions conversions
def load_dict_contractions():
    
    '''
    Dictionary of contractions as keys and their expanded words 
    as values.
    
    [Code modified from]: 
    https://stackoverflow.com/questions/19790188/expanding-english-\
    language-contractions-in-python
    '''
    
    return {
        "ain't": "is not",
        "amn't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "cuz": "because",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "could've": "could have",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "d'you": "do you",
        "e'er": "ever",
        "em": "them",
        "'em": "them",
        "everyone's": "everyone is",
        "finna": "fixing to",
        "gimme": "give me",
        "gonna": "going to",
        "gon't": "go not",
        "gotta": "got to",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how would",
        "how'll": "how will",
        "how're": "how are",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i'm'a": "i am about to",
        "i'm'o": "i am going to",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "i've": "i have",        
        "kinda": "kind of",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "may've": "may have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "might've": "might have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "must've": "must have",
        "needn't": "need not",
        "needn't've": "need not have",
        "ne'er": "never",
        "o'": "of",
        "o'clock": "of the clock",
        "o'er": "over",
        "ol'": "old",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",                
        "shalln't": "shall not",
        "shan't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "should've": "should have",
        "so's": "so as",
        "so've": "so have",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "something's": "something is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that'll": "that will",
        "that're": "that are",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there'll": "there will",
        "there're": "there are",
        "there's": "there is",
        "these're": "these are",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "this's": "this is",
        "those're": "those are",
        "to've": "to have",
        "'tis": "it is",
        "tis": "it is",
        "'twas": "it was",
        "twas": "it was",
        "wanna": "want to",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'd": "what did",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where're": "where are",
        "where's": "where is",
        "where've": "where have",
        "which's": "which is",
        "will've": "will have",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why've": "why have",
        "why's": "why is",
        "won't": "will not",
        "won't've": "will not have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "would've": "would have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
        }


# obtain POS tags
# NEED TO LOOK INTO THIS ONE!!!!!!!
def get_wordnet_pos(word):
    
    '''
    Function to map part-of-speech tag to first character 
    lemmatize() accepts.
    
    Input
    -----
    word : str
        Word to tag.
    
    Output
    ------
    tag_dict.get(tag, wordnet.NOUN)
       
    [Code taken from]:
    https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    '''
    
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)



# Apply text cleaning techniques
def clean_text(text, stop_words):
    '''Make text lowercase, tokenize words and words with apostrophes, convert contractions to full words,
    lemmatize by POS tag, remove stop words and words shorter than 3 letters.'''
    
    # make text lowercase
    text = text.lower().replace("’", "'")

    # initial tokenization to remove non-words
    tokenizer = RegexpTokenizer("([a-z]+(?:'[a-z]+)?)")
    words = tokenizer.tokenize(text)

    # convert contractions
    contractions = load_dict_contractions()
    words = [contractions[word] if word in contractions else word for word in words]
    text = ' '.join(words)

    # remove stop words, lemmatize using POS tags, and remove two-letter words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(text) \
             if word not in stop_words]
    
    # removing any words that got lemmatized into a stop word
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 2]
    text = ' '.join(words)
    
    return text

    
# perform multiple two-way t-tests
def two_way_tests(series_list):
    
    '''
    Function to run a two-sided t-test on each combination within
    a list of Pandas Series values.
    
    Returns a dictionary with the indices of the tested series as 
    the keys and the test results as the values.
    
    Input
    -----
    series_list : list (Pandas Series)
        A list in which each item is a Pandas Series of a 
        continuous variable within a subsection of your data.
        
        Example Series format : 
        ```df['*continuous_var'][df['*subsection'] == '*category']```
    
    Output
    ------
    compare_dict : dict
        
    '''
    compare_dict = {}
    for i in range(len(series_list)):
        count = i+1
        while count < len(series_list):
            compare_dict.update({(i,count): ttest_ind(series_list[i], series_list[count])})
            count += 1
    return compare_dict


# confusion matrix plotter
def plot_confusion_matrix(
        cm,
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues):
    
    '''
    This function prints and plots a model's confusion matrix.

    Input
    -----
    cm : sklearn confusion matrix
        `sklearn.metrics.confusion_matrix(y_true, y_pred)`
    classes : list (str)
        Names of target classes.

    Optional input
    --------------
    normalize : bool
        Whether to apply normalization (default=False).
        Normalization can be applied by setting `normalize=True`.
    title : str
        Title of the returned plot.
    cmap : matplotlib color map
        For options, visit:
        `https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html`

    Output
    ------
    Prints a stylized confusion matrix.


    [Code modified from work by Sean Abu Wilson.]
    '''

    # convert to percentage, if normalize set to True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # format true positives and others
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=15,
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # add axes labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    
# naive bayes feature importances printer
def print_nb_features(model, df, label_names, num_features=10):
    
    '''
    This function prints feature importances for Bernoulli and
    Multinomial Naive Bayes models, sorted by measure of importance.

    Input
    -----
    model : Naive Bayes model
        `sklearn.naive_bayes.BernoulliNB()`
        `sklearn.naive_bayes.MultinomialNB()`
    df : Pandas DataFrame
        Features used in model.

    Optional input
    --------------
    num_features : int
        The number of features to print (default=10).
        All feature importances can be shown by setting
        `num_features=df.shape[1]`.

    Output
    ------
    Prints labels and a list of features.
    '''
    
    # loop through each label
    for i,label in enumerate(label_names):
        # sorted features per class by importance
        prob_sorted = model.feature_log_prob_[i, :].argsort()

        # printout class's features
        print(f'{label.title()} tweets:\n{", ".join(list(np.take(df.columns, prob_sorted[:num_features])))}\n')
              

# decision tree feature importances plotter
def plot_tree_features(
        model,
        df,
        num_features=10,
        to_print=True,
        to_save=False,
        file_name=None):
    
    '''
    This function plots feature importances for Decision Tree models
    and optionally prints a list of tuples with features and their
    measure of importance.

    Input
    -----
    model : Decision Tree model
        `sklearn.tree.DecisionTreeClassifier()`
    df : Pandas DataFrame
        Features used in model.

    Optional input
    --------------
    num_features : int
        The number of features to plot/print (default=10).
        All feature importances can be shown by setting
        `num_features=df.shape[1]`.
    to_print : bool
        Whether to print list of feature names and their impurity
        decrease values (default=True).
        Printing can be turned off by setting `to_print=False`.
    file_name : str
        Path and name to save a graph (default=None).
        If `file_name=None`, the graph will not be saved.

    Output
    ------
    Prints a bar graph and optional list of tuples.
    '''

    features_dict = dict(zip(df.columns, model.feature_importances_))
    sorted_d = sorted(
        features_dict.items(),
        key=lambda x: x[1],
        reverse=True)[
        :num_features]

    # top 10 most important features
    tree_importance = [x[1] for x in sorted_d]

    # prettify the graph
    plt.figure(figsize=(12, 8))
    plt.title('Decision Tree Feature Importances', fontsize=25, pad=15)
    plt.xlabel('')
    plt.ylabel('Gini Importance', fontsize=22, labelpad=15)
    plt.ylim(bottom=sorted_d[-1][1]/1.75, top=sorted_d[0][1]*1.05)
    plt.xticks(rotation=60, fontsize=20)
    plt.yticks(fontsize=20)

    # plot
    plt.bar([x[0] for x in sorted_d], tree_importance)

    # prepare to display
    plt.tight_layout()

    if file_name:
        # save plot
        plt.savefig(file_name, bbox_inches='tight', transparent=True)

    # show plot
    plt.show()

    if to_print:
        # print a list of feature names and their impurity decrease value in
        # the decision tree
        print('\n\n\n')
        print(sorted_d)