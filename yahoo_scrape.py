# -*- coding: utf-8 -*-
"""
Created on Fri May 21 23:52:44 2021

@author: micud
"""
# 1. Install and Import Baseline Dependencies
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
from transformers import pipeline
import csv
import re
from time import sleep
import pandas as pd

# 2. Setup AI Model
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


headers = {
    'accept': '*/*',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.google.com',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
}

def get_article(card):
    """Extract article information from the raw html"""
    headline = card.find('h4', 's-title').text
    source = card.find("span", 's-source').text
    posted = card.find('span', 's-time').text.replace('·', '').strip()
    description = card.find('p', 's-desc').text.strip()
    raw_link = card.find('a').get('href')
    unquoted_link = requests.utils.unquote(raw_link)
    pattern = re.compile(r'RU=(.+)\/RK')
    clean_link = re.search(pattern, unquoted_link).group(1)
    
    article = (headline, source, posted, description, clean_link)
    return article

def get_the_news(search):
    """Run the main program"""
    template = 'https://news.search.yahoo.com/search?p={}'
    url = template.format(search)
    articles = []
    links = set()
    
    while True:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        cards = soup.find_all('div', 'NewsArticle')
        
        # extract articles from page
        for card in cards:
            article = get_article(card)
            link = article[-1]
            if not link in links:
                links.add(link)
                articles.append(article)        
                
        # find the next page
        try:
            url = soup.find('a', 'next').get('href')
            sleep(1)
        except AttributeError:
            break
            
    # save article data
    with open('results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Headline', 'Source', 'Posted', 'Description', 'Link'])
        writer.writerows(articles)
        
    return articles


watch_list = ['Ocado','Trainline']

master_df = []

for x in watch_list:

    raw_articles = get_the_news(x)
    
    
    story_df = pd.DataFrame(raw_articles, columns =['Headline', 'Source','Time','Snippet','URL'])[:3]
    
    
    # 4.3. Search and Scrape Cleaned URLs
    print('Scraping news links.')
    def scrape_and_process(URLs):
        ARTICLES = []
        for url in URLs:
            print('Reading: ',url)
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            results = soup.find_all('p')
            text = [res.text for res in results]
            words = ' '.join(text).split()[:350]
            ARTICLE = ' '.join(words)
            print(ARTICLE)
            ARTICLES.append(ARTICLE)
        return ARTICLES
    
    articles = scrape_and_process(list(story_df['URL'])) 
    
    
  
    
    # 4.4. Summarise all Articles
    print('Summarizing articles.')
    def summarize(articles):
        summaries = []
        for article in articles:
            input_ids = tokenizer.encode(article, return_tensors="pt")
            output = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            summaries.append(summary)
        return summaries
    
    summaries = summarize(articles)
    
    # 5. Adding Sentiment Analysis
    print('Calculating sentiment.')
    sentiment = pipeline("sentiment-analysis")
    scores = sentiment(summaries)
    
    model_results = pd.DataFrame(list(zip(articles,summaries,scores)),columns =['Full Article','AI Summary', 'Score'])
    
    model_results['Company'] = x
    
    final = story_df.merge(model_results,left_index=True,right_index=True,how='outer')
    
    master_df.append(final)
    
master_df = pd.concat(master_df)
master_df.to_excel("yahoo_results.xls")