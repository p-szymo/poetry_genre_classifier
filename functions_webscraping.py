# dataframe packages
import pandas as pd
import numpy as np

# webscraping packages
from selenium import webdriver
import requests as rq
from bs4 import BeautifulSoup as bs
import pytesseract

# string processing
import re
from ast import literal_eval
from unicodedata import normalize


# codes signifying genres within url
def load_genre_codes():
    
    '''
    Load a dictionary of codes used by PoetryFoundation.org in
    a genre page's URL.
    '''
    
    return {
        'augustan': 149,
        'beat': 150,
        'black_arts_movement': 304,
        'black_mountain': 151,
        'confessional': 152,
        'fugitive': 153,
        'georgian': 154,
        'harlem_renaissance': 155,
        'imagist': 156,
        'language_poetry': 157,
        'middle_english': 158,
        'modern': 159,
        'new_york_school': 160,
        'new_york_school_2nd_generation': 161,
        'objectivist': 162,
        'renaissance': 163,
        'romantic': 164,
        'victorian': 165
    }


# find urls for poets
def poet_urls_by_genre(genre_code, max_page_num=3):
    
    '''
    Function to scrape PoetryFoundation.org for URLs to the pages
    of each poet within a genre, represented by a genre code.

    Input
    -----
    genre_code : int
        Code within genre page's URL.
    max_page_num : int
        Number of pages to iterate through (default=3).
        As of now, there are no more than 3 pages of poets per 
        genre.

    Output
    ------
    poet_urls : list (str)
        List of URLs to each poet's page.
    
    NOTE: Selenium is known to encounter issues, sometimes causing
          the function to not work properly. Try re-running if 
          output is not as expected.
    '''

    # url requirements
    base_url = 'https://www.poetryfoundation.org/poets/browse#page='
    genre_addon = '&sort_by=last_name&school-period='

    # create empty list
    poet_urls = []
    # loop through desired number of pages
    for i in range(1, max_page_num+1):
        try:
            # instantiate a selenium browser
            driver = webdriver.Chrome()
            # load webpage
            driver.get(f'{base_url}{i}{genre_addon}{genre_code}')
            # find all links
            hrefs = driver.find_elements_by_xpath("//*[@href]")
            # find only links that match pattern for poet url
            pattern = re.compile(r'^.*/poets/(?!browse)[a-z\-]*$')
            poet_urls_by_page = [
                href.get_attribute('href') for href in hrefs if \
                pattern.match(href.get_attribute('href'))
            ]

            # only extend the list if there is something to extend
            if poet_urls_by_page:
                poet_urls.extend(poet_urls_by_page)
                # manually create some time between selenium browser, to
                # decrease chance of errors or IP block
                time.sleep(2.5)
            else:
                break
        # NOTE: a more specific except protocol may allow one to not have 
        # to re-run this function. One could re-run the try step until all 
        # possible links are grabbed
        except:
            break

    return poet_urls



# find urls for poems on poetryfoundation.org
def poem_urls_scraper(poet_url):
    
    '''
    Function to scrape PoetryFoundation.org for the URLs to a 
    poet's poems.

    Input
    -----
    poet_url : str
        URL to a poet's page.

    Output
    ------
    poem_text_urls : list (str)
        List of URLs for poems that are known to be text-based.
        
    poem_scan_urls : list (str)
        List of URLs for poems that are most likely scanned
        images.
        NOTE: It is possible that some of the poems associated
              with these URLs are text-based.
    '''

    # load a page and soupify it
    page = rq.get(poet_url)
    soup = bs(page.content, 'html.parser')

    # find all links that fit a certain pattern
    # finds links to poems that are text and easily scraped
    poems_text = soup.find_all(
        'a',
        href=re.compile('https://www.poetryfoundation.org/poems/[0-9]+/.*'),
        attrs={
            'class': None})
    # finds links to poems that are images
    poems_scan = soup.find_all(
        'a',
        href=re.compile('https://www.poetryfoundation.org/poetrymagazine/poems/[0-9]+/.*'),
        attrs={
            'class': None})

    # turn into lists
    if poems_text:
        poems_text_urls = [poem.get('href') for poem in poems_text]
    else:
        poems_text_urls = []

    if poems_scan:
        poems_scan_urls = [poem.get('href') for poem in poems_scan]
    else:
        poems_scan_urls = []

    # list of each set of urls, removing duplicates
    return list(set(poems_text_urls)), list(set(poems_scan_urls))


# scrape poems already in text format
def text_poem_scraper(poem_url):

    '''
    Function to scrape PoetryFoundation.org for the text of a poem.
    Returns a dictionary with poet name, poem url, poem title,
    poem as a list of lines, and poem as a single string.

    Input
    -----
    poem_url : str
        URL to a poem's page.

    Output
    ------
    info['poet'] : str
        Name of the poet.
        
    info['poem_url'] : str
        URL of the poem (same as the input).
            
    info['title'] : str
        Title of the poem.
    
    info['poem_lines'] : list (str)
        List of the lines of the poem, without any empty lines.
    
    info['poem_string'] : str
        Poem joined with newline characters.
    '''
    
    # load a page and soupify it
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')

    # scrape poet name
    poet = soup.find('a', href=re.compile('.*/poets/.*')).contents[0]

    # scrape title
    title = soup.find('h1').contents[-1].strip()

    # scrape poem
    try:
        # most frequent formatting
        lines_raw = soup.find_all(
            'div', {'style': 'text-indent: -1em; padding-left: 1em;'})
        # normalize text (from unicode)
        lines = [normalize('NFKD', str(line.contents[0]))
                 for line in lines_raw if line.contents]
        # remove some hanging html and left/right whitespace
        lines = [line.replace('<br/>', '').strip() for line in lines
                 if line.replace('<br/>', '').strip()]
        try:
            # remove certain bracket pattern (special cases)
            line_pattern = '>(.*?)<'
            lines = [re.search(line_pattern, line, re.I).group(
                1) if '<' in line else line for line in lines]
        except:
            try:
                # remove other bracket pattern (special cases)
                lines = [
                    re.sub('<.*>', '', line) if '<' in line \
                    else line for line in lines
                ]
            except:
                # else NaN
                lines = np.nan

    except:
        try:
            # if 'text-align' is justified
            lines_raw = soup.find_all('div', {'style': 'text-align: justify;'})
            # normalize text (from unicode)
            lines = [normalize('NFKD', str(line.contents[0]))
                     for line in lines_raw if line.contents]
            # remove some hanging html and left/right whitespace
            lines = [line.replace('<br/>', '').strip() for line in lines
                     if line.replace('<br/>', '').strip()]

            if lines == []:
                # if nothing grabbed
                try:
                    # scrape from PoemView
                    lines_raw = soup.find(
                        'div', {
                            'data-view': 'PoemView'}).get_text().split('\r')
                    # remove left/right whitespace
                    lines = [line.strip()
                             for line in lines_raw if line.strip()]
                except:
                    # else NaN
                    lines = np.nan
        except:
            # else NaN
            lines = np.nan

    # create string version of poem
    poem_string = '\n'.join(lines)

    # create and return dictionary
    info = {'poet': poet,
            'poem_url': poem_url,
            'title': title,
            'poem_lines': lines,
            'poem_string': poem_string}

    return info


# scrape text from poems in image format
def process_image(
        poem_url, 
        first=True, 
        poet=None, 
        title=None, 
        first_pattern='.*((?:\r?\n(?![A-HJ-Z][A-HJ-Z ][A-Z ]+$).*)*)',
        next_pattern='\n((?:\r?\n(?![A-HJ-Z][A-HJ-Z ][A-Z ]+$).*)*)'):
    
    '''
    Function to scrape PoetryFoundation.org single scanned image
    for the text in a poem. The process is different depending
    on whether or not the image is the first page of a poem.
    
    The optional parameters can be used when default patterns
    fail to scrape poems properly.
    
    Returns a dictionary with poet name, poem title, poem as a 
    list of lines, and the URL of the possible next page.

    Input
    -----
    poem_url : str
        URL to a poem's page.
        
    Optional input
    --------------
    first : bool
        Whether or not it is the first page of a poem
        (default=True).
        
    poet : str
        Name of the poet.
        
    title : str
        Title of the poem.
        
    first_pattern : str
        Regex pattern for scraping the first page of a poem.
        
    next_pattern : str
        Regex pattern for scraping the non-first pages of a poem.

    Output
    ------
    lines : list (str)
        List of lines of a poem.
        
    poet : str
        Name of the poet.
            
    title : str
        Title of the poem.
    
    next_page : str
        URL for the next page of the magazine.
    '''
    
    # load a page and soupify it
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    
    # first page process
    if first:
        
        # if no poet given, scrape it
        if not poet:
            poet = soup.find('a', href=re.compile('.*/poets/.*')).contents[0]
            
        # if no title given, scrape it
        if not title:
            # clean url
            jumble_pattern = r'-[0-9]+[a-z][0-9a-z]*$'
            clean_url = re.sub(jumble_pattern, '', poem_url)
            # regex title
            title_pattern = r'[a-z0-9\-]*$'
            title = re.search(
                title_pattern,
                clean_url,
                re.I).group().replace(
                '-',
                ' ').title()
            
    # find and save image to temporary file
    img_link = soup.find('img', src=re.compile('.*/jstor/.*'))['src']
    img_data = rq.get(img_link).content
    with open('data/temp.png', 'wb') as handle:
        handle.write(img_data)
        
    # scrape all text from image
    text = pytesseract.image_to_string('data/temp.png')

    # only capture poem, first page
    if first:
        try:
            # most common title format
            scan_pattern = fr'{title.split()[-1].upper()}{first_pattern}'
            lines = re.search(scan_pattern, text, re.MULTILINE).group(1).splitlines()
        except:
            try:
                # if only first letter capitalized
                scan_pattern = fr'{title.split()[-1]}{first_pattern}'
                lines = re.search(scan_pattern, text, re.MULTILINE).group(1).splitlines()
            except:
                try:
                    # if all lowercase
                    scan_pattern = fr'{title.split()[-1].lower()}{first_pattern}'
                    lines = re.search(scan_pattern, text, re.MULTILINE).group(1).splitlines()
                except:
                    # most common title format, first word
                    scan_pattern = fr'{title.split()[0].upper()}{first_pattern}'
                    lines = re.search(scan_pattern, text, re.MULTILINE).group(1).splitlines()

    # only capture poem, non-first pages
    else:
        scan_pattern = fr'{next_pattern}'
        lines = re.search(scan_pattern, text, re.MULTILINE).group(1).splitlines()
            
    # url for next pages
    next_page = soup.find(
        'a', attrs={
            'class': 'c-assetStack-media-control c-assetStack-media-control_next'})['href']

    # return all items for first page
    if first:
        return lines, poet, title, next_page

    # items for non-first pages
    return lines, next_page


# scrape text from poems in image format across possible multiple pages
def scan_poem_scraper(
        poem_url, 
        input_poet=None, 
        input_title=None, 
        first_pattern='.*((?:\r?\n(?![A-HJ-Z][A-HJ-Z ][A-Z ]+$).*)*)',
        next_pattern='\n((?:\r?\n(?![A-HJ-Z][A-HJ-Z ][A-Z ]+$).*)*)'):
    
    '''
    Function to scrape PoetryFoundation.org multiple scanned 
    images for the text in a poem.
    
    The optional parameters can be used when default patterns
    fail to scrape poems properly.
    
    Returns a dictionary with poet name, poem title, poem as a 
    list of lines, and the URL of the possible next page.

    Input
    -----
    poem_url : str
        URL to a poem's page.
        
    Optional input
    --------------    
    input_poet : str
        Name of the poet.
        
    input_title : str
        Title of the poem.
        
    first_pattern : str
        Regex pattern for scraping the first page of a poem.
        
    next_pattern : str
        Regex pattern for scraping the non-first pages of a poem.

    Output
    ------
    info['poet'] : str
        Name of the poet.
        
    info['poem_url'] : str
        URL of the poem (same as the input).
            
    info['title'] : str
        Title of the poem.
    
    info['poem_lines'] : list (str)
        List of the lines of the poem, without any empty lines.
    
    info['poem_string'] : str
        Poem joined with newline characters.
    '''
    
    # scrape first page
    lines, poet, title, next_page = process_image(poem_url, 
                                                  poet=input_poet, 
                                                  title=input_title, 
                                                  first_pattern=first_pattern,
                                                  next_pattern=next_pattern)
    
    # regex for bracketed or unbracketed page numbers
    page_number_pattern = r'[\[\(\{]?\s?[\d]+\s?[\]\)\}]?'
    
    # if the last line of poem text is a page number, keep scraping
    while re.match(page_number_pattern, lines[-1]):
        
        # scrape non-first pages
        add_lines, next_page = process_image(next_page, first=False, next_pattern=next_pattern)
        
        if not add_lines:
            break
        lines.extend(add_lines)

    # remove page numbers
    lines = [
        line for line in lines if line if not re.match(
            page_number_pattern, line)]

    # create string version of poem
    poem_string = '\n'.join(lines)

    # return dictionary
    info = {'poet': poet,
            'poem_url': poem_url,
            'title': title,
            'poem_lines': lines,
            'poem_string': poem_string}

    return info


# scrape poems in lesser-used text formats
def rescraper(poem_url, mode):
    
    '''
    Function to scrape PoetryFoundation.org for specific type 
    of text within a page's HTML objects.
    
    Returns a tuple with poem as a list of lines and poem as a
    single string.

    Input
    -----
    poem_url : str
        URL to a poem's page.
    mode : str
        The HTML object to be scraped.
        One of ['PoemView', 'poempara', 'p_all', 'justify', 
                'center'].
         
    Output
    ------
    lines_clean : list (str)
        List of the lines of the poem, without any empty lines.
    
    poem_string : str
        Poem joined with newline characters.
    '''
    
    # load a page and soupify it
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    
    if mode == 'PoemView':
        # scrape text from soup
        lines_raw = soup.find(
                            'div', {
                                'data-view': 'PoemView'}).get_text().split('\r')

        # initial process text
        lines = [normalize('NFKD', line).replace('\ufeff', '') 
                 for line in lines_raw if line]
        
    elif mode == 'poempara':
        # scrape text from soup
        lines_raw = soup.find_all('div', {'class': 'poempara'})

        # initial process text
        lines = [normalize('NFKD', str(line.contents[-1]))
                 for line in lines_raw if line.contents]
    
    elif mode == 'p_all':
        # scrape text from soup
        lines_raw = soup.find_all('p')[:-1]
        
        # initial process text
        lines = [normalize('NFKD', str(line.contents[0]))
                 for line in lines_raw if line]
        
    elif mode == 'justify':
        # scrape text from soup
        lines_raw = soup.find('div', {'style': 'text-align: justify;'}).contents

        # initial process text
        lines = [normalize('NFKD', str(line)) for line in lines_raw if line]
        
    elif mode == 'center':
        # scrape text from soup
        lines_raw = soup.find_all('div', {'style': 'text-align: center;'})
        
        # initial process text
        lines = [normalize('NFKD', str(line)) for line in lines_raw if line]
        
    # process text
    lines = [line.replace('<br/>', '') for line in lines]
    lines = [line.strip() for line in lines if line.strip()]
    line_pattern = '>(.*?)<'
    lines_clean = []
    for line in lines:
        if '<' in line:
            try:
                lines_clean.append(
                    re.search(
                        line_pattern,
                        line,
                        re.I).group(1).strip())
            except BaseException:
                continue
        else:
            lines_clean.append(line.strip())
            
    # create string version of poem
    poem_string = '\n'.join(lines_clean)
    
    return lines_clean, poem_string


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