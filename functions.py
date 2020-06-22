from selenium import webdriver
import re
import time

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