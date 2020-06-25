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

def poet_urls_by_genre(genre_code, max_page_num):
    '''Scraper for PoetryFoundation.org--scrapes urls for poets by genre.
       Input genre code and maximum number of pages to iterate through.
       Outputs a list of urls for each poet within the specified parameters.
       NOTE: Selenium is known to encounter issues, so sometimes this code does not work properly. Try re-running if output
             is not as expected.'''
    
    # url requirements
    base_url = 'https://www.poetryfoundation.org/poets/browse#page='
    genre_addon = '&sort_by=last_name&school-period='
    
    # create empty list
    poet_urls = []
    # loop through desired number of pages
    for i in range(1,max_page_num):
        try: 
            # instantiate a selenium browser
            driver = webdriver.Chrome()
            # load webpage
            driver.get(f'{base_url}{i}{genre_addon}{genre_code}')
            # find all links
            hrefs = driver.find_elements_by_xpath("//*[@href]")
            # find only links that match pattern for poet url
            pattern = re.compile('^.*/poets/(?!browse)[a-z\-]*$')
            poet_urls_by_page = [href.get_attribute('href') for href in hrefs if pattern.match(href.get_attribute('href'))]
            
            # only extend the list if there is something to extend
            if poet_urls_by_page:
                poet_urls.extend(poet_urls_by_page)
                # manually create some time between selenium browser, to decrease chance of errors or IP block
                time.sleep(2.5)
            else:
                break
        # NOTE: a more specific except protocol may allow one to not have to re-run this code, could re-run the code
        #       until all possible links are grabbed
        except:
            break
            
    return poet_urls

def poem_urls_scraper(poet_url):
    '''Scraper for PoetryFoundation.org--scrapes poem urls by poet.
       Input the url for a poet's page on PoetryFoundation.org.
       Output two lists: first, a list of urls for text poems; second, a list of urls for poems that are scans of the original
       magazine page.'''
    
    # load a page and soupify it
    page = rq.get(poet_url)
    soup = bs(page.content, 'html.parser')
    
    # find all links that fit a certain pattern
    # finds links to poems that are text and easily scraped
    poems_text = soup.find_all('a',
                               href=re.compile('https://www.poetryfoundation.org/poems/[0-9]+/.*'),
                               attrs={'class': None})
    # finds links to poems that are images
    poems_scan = soup.find_all('a',
                               href=re.compile('https://www.poetryfoundation.org/poetrymagazine/poems/[0-9]+/.*'),
                               attrs={'class': None})
    
    # turn into lists
    if poems_text:
        poems_text_urls = [poem.get('href') for poem in poems_text]
    else:
        poems_text_urls = []
        
    if poems_scan:
        poems_scan_urls = [poem.get('href') for poem in poems_scan]
    else:
        poems_scan_urls = []
    
    return poems_text_urls, poems_scan_urls

def poem_scraper(poem_url):
    '''Scraper for PoetryFoundation.org--scrapes poet name, poem title, poem year, list of poem's lines,
       and the poem as a string.
       Input the url for a poem's page on PoetryFoundation.org.
       Output is a list.'''
    
    # load a page and soupify it
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    
    # series of try/except statements to scrape info or return NaN value if desired info cannot be scraped
    try:
        poet = soup.find('a', href=re.compile('.*/poets/.*')).contents[0]
    except:
        poet = np.nan
        
    try:
        title = soup.find('h1').contents[-1].strip()
    except:
        try:
            title_pattern = '[a-z\-]*$'
            title = re.search(title_pattern, poem_url, re.I).group().replace('-', ' ').title()
        except:
            title = np.nan
        
    try:
        lines_raw = soup.find_all('div', {'style': 'text-indent: -1em; padding-left: 1em;'})
        lines = [normalize('NFKD', str(line.contents[0])) for line in lines_raw if line.contents]
        lines = [line.replace('<br/>', '') for line in lines]
        try:
            line_pattern = '>(.*?)<'
            lines = [re.search(line_pattern, line, re.I).group(1) if '<' in line else line for line in lines]
        except:
            try:
                lines = [re.sub('<.*>', '', line) if '<' in line else line for line in lines]
            except:
                lines = np.nan
        if lines == []:
            try:
                img_link = soup.find('img', src=re.compile('.*/jstor/.*'))['src']
                img_data = rq.get(img_link).content
                with open('poem_imgs/temp.png', 'wb') as handle:
                    handle.write(img_data)
                text = pytesseract.image_to_string('poem_imgs/temp.png')
                scan_pattern = fr'{title.upper()}\s*((.*\s.*)*)'
                lines = re.search(scan_pattern, text, re.I).group(1).splitlines()
            except:
                lines = np.nan
        lines = [line.strip() for line in lines if line]
    except:
        try:
            lines_raw = soup.find_all('div', {'style': 'text-align: justify;'})
            lines = [normalize('NFKD', str(line.contents[0])) for line in lines_raw if line.contents]
            lines = [line.replace('<br/>', '') for line in lines]
            lines = [line.strip() for line in lines if line]
            if lines == []:
                try:
                    lines_raw = soup.find('div', {'data-view': 'PoemView'}).contents[1]
                    lines = [normalize('NFKD', str(line)) for line in lines_raw if line]
                    lines = [line.replace('<br/>', '') for line in lines]
                    lines = [line.strip() for line in lines if line]
                except:
                    lines = np.nan
        except:
            lines = np.nan
            
        
    try:
        poem_string = '\n'.join(lines)
    except:
        poem_string = np.nan
        
    try:
        year_blurb = soup.find('span', {'class': 'c-txt c-txt_note c-txt_note_mini'}).contents[2]
        year_pattern = r'[12]\d{3}'
        year = int(re.search(year_pattern, year_blurb, re.I).group())
    except:
        try:
            year_blurb = soup.find_all('span', {'class': 'c-txt c-txt_note c-txt_note_mini'})[-1].contents[2]
            year_pattern = r'[12]\d{3}'
            year = int(re.search(year_pattern, year_blurb, re.I).group())
        except:
            year = np.nan
    
    info = [poet, title, year, lines, poem_string]
    
    return info

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
def poempara_rescraper(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all('div', {'class': 'poempara'})
    lines = [normalize('NFKD', str(line.contents[-1])) for line in lines_raw if line.contents]
    lines = [line.replace('<br/>', '') for line in lines]
    line_pattern = '>(.*?)<'
    lines = [re.search(line_pattern, line, re.I).group(1) if '<' in line else line for line in lines]
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
                lines_clean.append(re.search(line_pattern, line, re.I).group(1).strip())
            except:
                continue
        else:
            lines_clean.append(line.strip())
    poem_string = '\n'.join(lines_clean)
    return lines_clean, poem_string

def ranged_rescraper(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all('div', {'style': 'text-indent: -1em; padding-left: 1em;'})
    lines = [normalize('NFKD', str(line.contents[-1])) for line in lines_raw if line.contents]
    lines = [line.replace('<br/>', '') for line in lines]
    line_pattern = '>(.*?)<'
    lines = [re.search(line_pattern, line, re.I).group(1) if '<' in line else line for line in lines]
    lines = [line.strip() for line in lines if line]
    poem_string = '\n'.join(lines)
    return lines, poem_string

def modified_regular_rescraper(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all('div', {'style': 'text-indent: -1em; padding-left: 1em;'})[0]
    lines = [normalize('NFKD', str(line)) for line in lines_raw if line]
    lines = [line.replace('<br/>', '') for line in lines]
    lines = [line.strip() for line in lines if line.strip()]
    line_pattern = '>(.*?)<'
    lines_clean = []
    for line in lines:
        if '<' in line:
            try:
                lines_clean.append(re.search(line_pattern, line, re.I).group(1).strip())
            except:
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
                lines_clean.append(re.search(line_pattern, line, re.I).group(1).strip())
            except:
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
                lines_clean.append(re.search(line_pattern, line, re.I).group(1).strip())
            except:
                continue
        else:
            lines_clean.append(line.strip())
    poem_string = '\n'.join(lines_clean)
    return lines_clean, poem_string

def p_rescraper_all(poem_url):
    page = rq.get(poem_url)
    soup = bs(page.content, 'html.parser')
    lines_raw = soup.find_all('p')[:-1]
    lines = [normalize('NFKD', str(line.contents[0])) for line in lines_raw if line]
    lines = [line.replace('<br/>', '') for line in lines]
    lines = [line.strip() for line in lines if line.strip()]
    line_pattern = '>(.*?)<'
    lines_clean = []
    for line in lines:
        if '<' in line:
            try:
                lines_clean.append(re.search(line_pattern, line, re.I).group(1).strip())
            except:
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
                lines_clean.append(re.search(line_pattern, line, re.I).group(1).strip())
            except:
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
    except:
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
    except:
        try:
            scan_pattern = fr'{title.split()[-1].upper()}\s*((.*\s.*)*)'
            lines = re.search(scan_pattern, text, re.I).group(1).splitlines()
            lines = [line.strip() for line in lines if line]
        except:
            scan_pattern = fr'{title.split()[0].upper()}\s*((.*\s.*)*)'
            lines = re.search(scan_pattern, text, re.I).group(1).splitlines()
            lines = [line.strip() for line in lines if line]
    poem_string = '\n'.join(lines)
    return lines, poem_string