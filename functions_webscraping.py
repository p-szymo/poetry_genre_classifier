import pandas as pd
import numpy as np

from selenium import webdriver
import re
import time
import requests as rq
from bs4 import BeautifulSoup as bs
from unicodedata import normalize
import pytesseract
from ast import literal_eval


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


def process_image(poem_url, first=True):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    if first:
        poet = soup.find('a', href=re.compile('.*/poets/.*')).contents[0]
        jumble_pattern = r'-[0-9]+[a-z][0-9a-z]*$'
        clean_url = re.sub(jumble_pattern, '', poem_url)
        title_pattern = r'[a-z0-9\-]*$'
        title = re.search(
            title_pattern,
            clean_url,
            re.I).group().replace(
            '-',
            ' ').title()
    img_link = soup.find('img', src=re.compile('.*/jstor/.*'))['src']
    img_data = rq.get(img_link).content
    with open('data/temp.png', 'wb') as handle:
        handle.write(img_data)
    text = pytesseract.image_to_string('data/temp.png')
    if first:
        scan_pattern = fr'{title.split()[-1].upper()}\b.*((\r?\n(?![A-Z][A-Z ]{3,}$).*)*)'
    else:
        scan_pattern = r'\n((\r?\n(?![A-Z][A-Z ]{3,}$).*)*)'
    lines = re.search(scan_pattern, text, re.MULTILINE).group(1).splitlines()
    next_page = soup.find(
        'a', attrs={
            'class': 'c-assetStack-media-control c-assetStack-media-control_next'})['href']

    if first:
        return lines, poet, title, next_page

    return lines, next_page


def scan_poem_scraper(poem_url):
    lines, poet, title, next_page = process_image(poem_url)
    while re.match(r'[\[\(]?[\d]+[\]\)]?', lines[-1]):
        add_lines, next_page = process_image(next_page, first=False)
        if not add_lines:
            break
        lines.extend(add_lines)

    lines = [
        line for line in lines if line if not re.match(
            r'[\[\(]?[\d]+[\]\)]?', line)]

    # create string version of poem
    poem_string = '\n'.join(lines)

    info = {'poet': poet,
            'poem_url': poem_url,
            'title': title,
            'poem_lines': lines,
            'poem_string': poem_string}

    return info


def text_poem_scraper_BACKUP_UNPARED(poem_url):

    # load a page and soupify it
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')

    # scrape poet name
    try:
        # from web page
        poet = soup.find('a', href=re.compile('.*/poets/.*')).contents[0]
    except BaseException:
        # else NaN if an error occurs
        poet = np.nan

    # scrape title
    try:
        # from web page
        title = soup.find('h1').contents[-1].strip()
    except BaseException:
        try:
            # else from url
            title_pattern = r'[a-z\-]*$'
            title = re.search(
                title_pattern,
                poem_url,
                re.I).group().replace(
                '-',
                ' ').title()
        except BaseException:
            # else NaN if an error occurs
            title = np.nan

    # scrape poem
    try:
        # most frequent formatting
        lines_raw = soup.find_all(
            'div', {'style': 'text-indent: -1em; padding-left: 1em;'})
        # normalize text (from unicode)
        lines = [normalize('NFKD', str(line.contents[0]))
                 for line in lines_raw if line.contents]
        # remove some hanging html
        lines = [line.replace('<br/>', '') for line in lines]
        try:
            # remove certain bracket pattern (special cases)
            line_pattern = '>(.*?)<'
            lines = [re.search(line_pattern, line, re.I).group(
                1) if '<' in line else line for line in lines]
        except BaseException:
            try:
                # remove other bracket pattern (special cases)
                lines = [
                    re.sub(
                        '<.*>',
                        '',
                        line) if '<' in line else line for line in lines]
            except BaseException:
                # else NaN
                lines = np.nan

    except BaseException:
        try:
            # if 'text-align' is justified
            lines_raw = soup.find_all('div', {'style': 'text-align: justify;'})
            # normalize text (from unicode)
            lines = [normalize('NFKD', str(line.contents[0]))
                     for line in lines_raw if line.contents]
            # remove some hanging html
            lines = [line.replace('<br/>', '') for line in lines]
            # remove left/right whitespace
            lines = [line.strip() for line in lines if line]

            if lines == []:
                # if nothing grabbed
                try:
                    # scrape 'PoemView' html type
                    lines_raw = soup.find(
                        'div', {'data-view': 'PoemView'}).contents[1]
                    # normalize text (from unicode)
                    lines = [normalize('NFKD', str(line))
                             for line in lines_raw if line]
                    lines = [line.replace('<br/>', '') for line in lines]
                    lines = [line.strip() for line in lines if line]
                except BaseException:
                    lines = np.nan
        except BaseException:
            lines = np.nan

    try:
        poem_string = '\n'.join(lines)
    except BaseException:
        poem_string = np.nan

    try:
        year_blurb = soup.find(
            'span', {
                'class': 'c-txt c-txt_note c-txt_note_mini'}).contents[2]
        year_pattern = r'[12]\d{3}'
        year = int(re.search(year_pattern, year_blurb, re.I).group())
    except BaseException:
        try:
            year_blurb = soup.find_all(
                'span', {'class': 'c-txt c-txt_note c-txt_note_mini'})[-1].contents[2]
            year_pattern = r'[12]\d{3}'
            year = int(re.search(year_pattern, year_blurb, re.I).group())
        except BaseException:
            year = np.nan

    info = [poet, title, year, lines, poem_string]

    return info


# scrape poems for text and other info on poetryfoundation.org
def poem_scraper(poem_url):
    '''Scraper for PoetryFoundation.org--scrapes poet name, poem title, poem year, list of poem's lines,
       and the poem as a string.
       Input the url for a poem's page on PoetryFoundation.org.
       Output is a list.'''

    # load a page and soupify it
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')

    # series of try/except statements to scrape info or return NaN value if
    # desired info cannot be scraped
    try:
        poet = soup.find('a', href=re.compile('.*/poets/.*')).contents[0]
    except BaseException:
        poet = np.nan

    try:
        title = soup.find('h1').contents[-1].strip()
    except BaseException:
        try:
            title_pattern = r'[a-z\-]*$'
            title = re.search(
                title_pattern,
                poem_url,
                re.I).group().replace(
                '-',
                ' ').title()
        except BaseException:
            title = np.nan

    try:
        lines_raw = soup.find_all(
            'div', {'style': 'text-indent: -1em; padding-left: 1em;'})
        lines = [normalize('NFKD', str(line.contents[0]))
                 for line in lines_raw if line.contents]
        lines = [line.replace('<br/>', '') for line in lines]
        try:
            line_pattern = '>(.*?)<'
            lines = [re.search(line_pattern, line, re.I).group(
                1) if '<' in line else line for line in lines]
        except BaseException:
            try:
                lines = [
                    re.sub(
                        '<.*>',
                        '',
                        line) if '<' in line else line for line in lines]
            except BaseException:
                lines = np.nan
        if lines == []:
            try:
                img_link = soup.find(
                    'img', src=re.compile('.*/jstor/.*'))['src']
                img_data = rq.get(img_link).content
                with open('poem_imgs/temp.png', 'wb') as handle:
                    handle.write(img_data)
                text = pytesseract.image_to_string('poem_imgs/temp.png')
                scan_pattern = fr'{title.upper()}\s*((.*\s.*)*)'
                lines = re.search(
                    scan_pattern, text, re.I).group(1).splitlines()
            except BaseException:
                lines = np.nan
        lines = [line.strip() for line in lines if line]
    except BaseException:
        try:
            lines_raw = soup.find_all('div', {'style': 'text-align: justify;'})
            lines = [normalize('NFKD', str(line.contents[0]))
                     for line in lines_raw if line.contents]
            lines = [line.replace('<br/>', '') for line in lines]
            lines = [line.strip() for line in lines if line]
            if lines == []:
                try:
                    lines_raw = soup.find(
                        'div', {'data-view': 'PoemView'}).contents[1]
                    lines = [normalize('NFKD', str(line))
                             for line in lines_raw if line]
                    lines = [line.replace('<br/>', '') for line in lines]
                    lines = [line.strip() for line in lines if line]
                except BaseException:
                    lines = np.nan
        except BaseException:
            lines = np.nan

    try:
        poem_string = '\n'.join(lines)
    except BaseException:
        poem_string = np.nan

    try:
        year_blurb = soup.find(
            'span', {
                'class': 'c-txt c-txt_note c-txt_note_mini'}).contents[2]
        year_pattern = r'[12]\d{3}'
        year = int(re.search(year_pattern, year_blurb, re.I).group())
    except BaseException:
        try:
            year_blurb = soup.find_all(
                'span', {'class': 'c-txt c-txt_note c-txt_note_mini'})[-1].contents[2]
            year_pattern = r'[12]\d{3}'
            year = int(re.search(year_pattern, year_blurb, re.I).group())
        except BaseException:
            year = np.nan

    info = [poet, title, year, lines, poem_string]

    return info

# combines scraping of poetry in a loop and creates a dataframe


def pf_scraper(poet_urls_dict, genre, time_sleep=1):
    '''Scraper for PoetryFoundation.org--scrapes poet name, poem title, poem year, list of poem's lines,
       and the poem as a string.
       Input is a dictionary with genres as keys and urls to poets' pages as values, as well as the genre you wish to scrape.
           Designed to be used in a loop, so if there is an error along the way, you could feasibly have some progress saved.
       Optional input of time to sleep between loop.
       Output is a Pandas DataFrame.'''

    # instantiate an empty list
    ultra_list = []

    # set up a for loop to iterate through urls in genre
    for poet_url in poet_urls_dict[genre]:

        # scrape urls for both types of pages, text poems and scanned poems
        poem_text_urls, poem_scan_urls = poem_urls_scraper(poet_url)

        # instantiate a list with url and genre info, then scrape the rest of the info using earlier function,
        # then add it to the list that will get converted into a dataframe
        for poem_url in poem_text_urls:
            info = [poet_url, genre, poem_url]
            info.extend(poem_scraper(poem_url))
            ultra_list.append(info)

        for poem_url in poem_scan_urls:
            info = [poet_url, genre, poem_url]
            info.extend(poem_scraper(poem_url))
            ultra_list.append(info)

        # pause the for loop for a second to try to prevent being blocked
        time.sleep(time_sleep)

    df = pd.DataFrame(ultra_list)

    return df

# BELOW IS A SERIES OF RESCRAPER FUNCTIONS FOR SPECIFIC CASES THAT SHOULD IDEALLY BE BUILT INTO THE LARGER FUNCTION ABOVE
# DOCUMENTATION AND COMMENTS ABSENT DUE TO LACK OF TIME


def poempara_rescraper(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all('div', {'class': 'poempara'})
    lines = [normalize('NFKD', str(line.contents[-1]))
             for line in lines_raw if line.contents]
    lines = [line.replace('<br/>', '') for line in lines]
    line_pattern = '>(.*?)<'
    lines = [re.search(line_pattern, line, re.I).group(
        1) if '<' in line else line for line in lines]
    lines = [line.strip() for line in lines if line]
    poem_string = '\n'.join(lines)
    return lines, poem_string


def PoemView_rescraper(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find('div', {'data-view': 'PoemView'}).contents
    lines = [normalize('NFKD', str(line)) for line in lines_raw if line]
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
    poem_string = '\n'.join(lines_clean)
    return lines_clean, poem_string


def ranged_rescraper(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all(
        'div', {'style': 'text-indent: -1em; padding-left: 1em;'})
    lines = [normalize('NFKD', str(line.contents[-1]))
             for line in lines_raw if line.contents]
    lines = [line.replace('<br/>', '') for line in lines]
    line_pattern = '>(.*?)<'
    lines = [re.search(line_pattern, line, re.I).group(
        1) if '<' in line else line for line in lines]
    lines = [line.strip() for line in lines if line]
    poem_string = '\n'.join(lines)
    return lines, poem_string


def modified_regular_rescraper(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all(
        'div', {'style': 'text-indent: -1em; padding-left: 1em;'})[0]
    lines = [normalize('NFKD', str(line)) for line in lines_raw if line]
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
    poem_string = '\n'.join(lines_clean)
    return lines_clean, poem_string


def justify_rescraper(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find('div', {'style': 'text-align: justify;'}).contents
    lines = [normalize('NFKD', str(line)) for line in lines_raw if line]
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
    poem_string = '\n'.join(lines_clean)
    return lines_clean, poem_string


def center_rescraper(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all('div', {'style': 'text-align: center;'})
    lines = [normalize('NFKD', str(line)) for line in lines_raw if line]
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
    poem_string = '\n'.join(lines_clean)
    return lines_clean, poem_string


def p_rescraper_all(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all('p')[:-1]
    lines = [normalize('NFKD', str(line.contents[0]))
             for line in lines_raw if line]
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
    poem_string = '\n'.join(lines_clean)
    return lines_clean, poem_string


def p_rescraper_2(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all('p')[2].contents
    lines = [normalize('NFKD', str(line)) for line in lines_raw if line]
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
    poem_string = '\n'.join(lines_clean)
    return lines_clean, poem_string


def image_rescraper_POETRY(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    img_link = soup.find('img', src=re.compile('.*/jstor/.*'))['src']
    img_data = rq.get(img_link).content
    with open('poem_imgs/temp.png', 'wb') as handle:
        handle.write(img_data)
    text = pytesseract.image_to_string('poem_imgs/temp.png')
    scan_pattern = r'POETRY\s*((.*\s.*)*)'
    lines = re.search(scan_pattern, text, re.I).group(1).splitlines()
    lines = [line.strip() for line in lines if line]
    poem_string = '\n'.join(lines)
    return lines, poem_string


def image_rescraper_poet(poem_url, poet):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    img_link = soup.find('img', src=re.compile('.*/jstor/.*'))['src']
    img_data = rq.get(img_link).content
    with open('poem_imgs/temp.png', 'wb') as handle:
        handle.write(img_data)
    text = pytesseract.image_to_string('poem_imgs/temp.png')
    try:
        scan_pattern = fr'{poet.upper()}\s*((.*\s.*)*)'
        lines = re.search(scan_pattern, text, re.I).group(1).splitlines()
        lines = [line.strip() for line in lines if line]
    except BaseException:
        scan_pattern = fr'{poet.split()[0].upper()}\s*((.*\s.*)*)'
        lines = re.search(scan_pattern, text, re.I).group(1).splitlines()
        lines = [line.strip() for line in lines if line]
    poem_string = '\n'.join(lines)
    return lines, poem_string


def image_rescraper_title(poem_url, title):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    img_link = soup.find('img', src=re.compile('.*/jstor/.*'))['src']
    img_data = rq.get(img_link).content
    with open('poem_imgs/temp.png', 'wb') as handle:
        handle.write(img_data)
    text = pytesseract.image_to_string('poem_imgs/temp.png')
    try:
        scan_pattern = fr'{title.upper()}\s*((.*\s.*)*)'
        lines = re.search(scan_pattern, text, re.I).group(1).splitlines()
        lines = [line.strip() for line in lines if line]
    except BaseException:
        try:
            scan_pattern = fr'{title.split()[-1].upper()}\s*((.*\s.*)*)'
            lines = re.search(scan_pattern, text, re.I).group(1).splitlines()
            lines = [line.strip() for line in lines if line]
        except BaseException:
            scan_pattern = fr'{title.split()[0].upper()}\s*((.*\s.*)*)'
            lines = re.search(scan_pattern, text, re.I).group(1).splitlines()
            lines = [line.strip() for line in lines if line]
    poem_string = '\n'.join(lines)
    return lines, poem_string
