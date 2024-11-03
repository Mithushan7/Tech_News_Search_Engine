import os
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin, urlparse

# Global set to store unique URLs
visited_urls = set()
output_dir = "crawled_pages"

# Function to set up the Selenium WebDriver
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (without opening the browser)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def crawl(seed_url, max_pages=250, delay=1, output_dir=output_dir):
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the queue with the seed URL and set the page counter to 0
    queue = [seed_url]
    page_count = 0

    driver = setup_driver()

    # Loop while there are URLs in the queue and the page limit hasn't been reached
    while queue and page_count < max_pages:
        url = queue.pop(0)

        if url not in visited_urls:
            try:
                driver.get(url)
                time.sleep(delay)  # Allow time for the JavaScript to load

                content = driver.page_source  # Get the rendered page source
                save_page(url, content, output_dir)

                visited_urls.add(url)
                page_count += 1
                print(f"Page {page_count}: {url} crawled successfully.")

                # Extract links from the rendered page
                links = driver.find_elements(By.TAG_NAME, "a")
                for link in links:
                    absolute_url = urljoin(url, link.get_attribute('href'))
                    if is_valid(absolute_url, seed_url) and absolute_url not in visited_urls:
                        queue.append(absolute_url)

            except Exception as e:
                print(f"Failed to crawl {url}: {e}")

    driver.quit()

def save_page(url, content, output_dir):
    # Extract the page name from the URL to use as the filename
    page_name = url.split("/")[-1] if url.split("/")[-1] else "index"
    file_path = os.path.join(output_dir, f"{page_name}.txt")

    # Save the content to a text file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Saved: {file_path}")

def is_valid(url, seed_url):
    return urlparse(url).netloc == urlparse(seed_url).netloc

if __name__ == "__main__":
    seed_page = "https://www.theverge.com/tech"  # Replace with the actual seed page
    crawl(seed_page, max_pages=250, delay=1)
