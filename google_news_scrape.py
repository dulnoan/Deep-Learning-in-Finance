# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:32:48 2021

@author: micud

https://github.com/kotartemiy/pygooglenews

"""
import pandas as pd
from pygooglenews import GoogleNews
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
from transformers import pipeline



# 2. Setup AI Model
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)



gn = GoogleNews(country = 'UK')

watchlist = ['IQE','Trainline','Sumo Digital']



master_df = []

for company_name in watchlist:
    print("\n\nWebscraping: ", company_name,"\n")
    story_list = []
    story_df = []
    search = gn.search(f'intitle:{company_name}', when = '2d')
    
    
    for item in search['entries']:
        story_info = []
        story_info = {'Headline': item.title , 'URL': item.link}
        story_info['Company'] = company_name
        story_list.append(story_info)
        story_df = pd.DataFrame(story_list, columns =['Company','Headline', 'URL'])


    
    # 4.3. Search and Scrape Cleaned URLs
    print('Scraping news links.')
    def scrape_and_process(URLs):
        ARTICLES = []
        for url in URLs:
            #print('Reading: ',url)
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            results = soup.find_all('p')
            text = [res.text for res in results]
            words = ' '.join(text).split()[:300]
            ARTICLE = ' '.join(words)
            ARTICLES.append(ARTICLE)
        return ARTICLES
    
    articles = scrape_and_process(list(story_df['URL'])) 
    
    
  
    
    # 4.4. Summarise all Articles
    print('Summarizing articles.')
    def summarize(articles):
        summaries = []
        for article in articles:
            input_ids = tokenizer.encode(article, return_tensors="pt")
            output = model.generate(input_ids, max_length=100,min_length=5,num_beams=5, early_stopping=True)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            summaries.append(summary)
        return summaries
    
    summaries = summarize(articles)
    
    # 5. Adding Sentiment Analysis
    print('Calculating sentiment.')
    scores = []
    sentiment = pipeline("sentiment-analysis")
    scores = sentiment(summaries)
    
    model_results = []
    model_results = pd.DataFrame(list(zip(articles,summaries,scores)),columns =['Full Article','AI Summary', 'Sentiment Score'])

    
    final = []
    final = story_df.merge(model_results,left_index=True,right_index=True,how='outer')
    
    master_df.append(final)
    
master_df = pd.concat(master_df)
master_df.to_excel("google_results.xls")

