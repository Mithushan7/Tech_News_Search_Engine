import time
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

#Global list to avoid recrawling visited links
visited_url= []

#ethical crawling variable 
#setting limiting factor variables in one place for easy changes
file_count=1
max_pages_save = 400 #code will stop after reaching 400 pages
max_unique_link = 1000 
delay_requests = 5 #wait 5 second between requests 

#list to avoid crawling ceratin links and anchor text
#this is done to capture more diverse documents 
#or else most of the crawl pages are only from the below listed url sets and anchor texts
excluded_keywords = ["comments", "privacy policy", "podcast", "applepodcast", "video", "advertisement", "terms", "cookie", "service"]
#once the program goes into apple podcast hyperlinks 
#its only crawling the poscast download pages effecting the required diverse set of documents
excluded_domains = ["podcasts.apple.com", "apple.com"]

#function to save the parsed webpages as text files
def save_pages(anchor_text,url,content):
    global file_count

    #creating if and for loop to check for specific keywords as anchor text
    # is_it_containing_excluded_keyword = False
    # for keyword in excluded_keywords:
    #     if keyword in anchor_text.strip().lower():
    #         is_it_containing_excluded_keyword = True
    #         return
    #function will exit if keyword in found as the anchor text

    #page number counter to save the program
    filename =  f"page{file_count}.txt"
    file_count = file_count+1
    #creating folder to save scraped pages 
    if not os.path.exists('crawled_pages'):
        os.makedirs('crawled_pages')
    
    file_path = os.path.join('crawled_pages',filename)

    #adding the title as the anchor text and attaching the url 
    #removing empty white line in the content
    content = f"URL: {url}\nTitle: {anchor_text.strip()}\n{content}"

    with open(file_path,'w',encoding='utf-8') as file:
        file.write('\n'.join([line for line in content.splitlines() if line.strip()]))

#function to check for listed URL to skip
def excluded_url_checker (url):
    for domain in excluded_domains:
        if domain in url:
            return True #if dmoain is found it will return true
    
    return False #if domain is not in the list it will return false

#main function to crawl the pages main requirment 
def crawl_page(url, depth=0, max_depth=20, retries=2):
    global file_count

    #condition loop to check for limit of allowed crawled pages
    #along with it, the loop check for max allowed unique links and max_depth of BFS
    if file_count > max_pages_save or len(visited_url)>= max_unique_link or depth > max_depth:
        return
    
    #error handling with 2 retries 
    for attempt in range(retries):

        try:

            response = requests.get(url)
            response.raise_for_status()

            soup= BeautifulSoup (response.text,'html.parser')
            anchors = soup.find_all('a')

            #loop to parse the a tag link for text
            for anchor in anchors:
                link = anchor.get('href')
                anchor_text = anchor.get_text(strip=True)

                if link and anchor_text:

                    #converting relative link to full link
                    full_url =  urljoin(url, link)

                    if full_url not in visited_url:

                        is_it_containing_excluded_keyword = False
                        for keyword in excluded_keywords:
                            if keyword in anchor_text.strip().lower():
                                is_it_containing_excluded_keyword =True
                                break
                        
                        if not is_it_containing_excluded_keyword and not excluded_url_checker(full_url):
                            visited_url.append(full_url)

                            if file_count <= max_pages_save:
                                save_pages(anchor_text,full_url,soup.get_text())
                            
                            if file_count > max_pages_save:
                                print ("reached max limit")
                                return
                            
                            crawl_page(full_url,depth+1)
            time.sleep(delay_requests)

            break

        except requests.RequestException as e:
            print(f"Error crawling {url} on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                print(f"Failed to crawl {url} after {retries} attempts.")


def start_crawling(seed_url):
    crawl_page(seed_url)

def save_global_url_list ():
    with open('visited_url,txt','w',encoding='utf-8') as file:
        file.write("\n".join(visited_url))

seed_url = 'https://www.theverge.com/tech'

start_crawling(seed_url)
save_global_url_list()
