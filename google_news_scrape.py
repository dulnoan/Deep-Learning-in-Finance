# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:32:48 2021

@author: micud

Key packages used:
pygooglenews = reliable package to scrape URL results to overcome beautiful soup issues on my end. https://github.com/kotartemiy/pygooglenews
beautifulsoup = used to scrape text in webpages
financial-summarization-pegasus = Deep Learning model that has been pretrained specifically for Financial summarization

"""

import pandas as pd
from bs4 import BeautifulSoup
import requests
from pygooglenews import GoogleNews
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import pipeline

#Specify Watchlist
watchlist = ['IQE','Trainline','Sumo Digital']

#Specify Region
gn = GoogleNews(country = 'UK')

#Setup AI Model using Pegasus which has been trained for Financial Summarization
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

#This will be the final dataframe output where results for each company will be appended
master_df = []

for company_name in watchlist:

    story_list = []
    story_df = []
    
    #Find results with "company_name" in title
    search = gn.search(f'intitle:{company_name}', when = '2d')
    
    print("\n\nWebscraping: ", company_name,"\n")
    
    
    #Loop through dictionary 'search entries' to get Headline, URL etc and convert to a dataframe
    for item in search['entries']:
        story_info = []
        story_info = {'Headline': item.title , 'URL': item.link}
        story_info['Company'] = company_name
        story_list.append(story_info)
        story_df = pd.DataFrame(story_list, columns =['Company','Headline', 'URL'])


    
    #Scrape articles from dataframe list story_df['URL']. Articles are limited to 300 words (Model/Hardware Limitation).
    print(f'Scraping {company_name} news links.')
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
 
    
    #Summarise all Articles by passing in list of articles scraped.
    print(f'Summarizing {company_name} articles.')
    def summarize(articles):
        summaries = []
        for article in articles:
            input_ids = tokenizer.encode(article, return_tensors="pt")
            output = model.generate(input_ids, max_length=100,min_length=5,num_beams=5, early_stopping=True)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            summaries.append(summary)
        return summaries
    
    summaries = summarize(articles)
    
    
    #Apply Sentiment Analysis
    print(f'Calculating {company_name} sentiment.')
    scores = []
    sentiment = pipeline("sentiment-analysis")
    scores = sentiment(summaries)
    
    
    #Create new dataframe model_results for full & summarised text and sentiment score.
    model_results = []
    model_results = pd.DataFrame(list(zip(articles,summaries,scores)),columns =['Full Article','AI Summary', 'Sentiment Score'])
    

    #Combined web scraped information with model results into a new combined dataframe 
    combined = []
    combined = story_df.merge(model_results,left_index=True,right_index=True,how='outer')
    
    #Append to master_df with results from other companies. 
    master_df.append(combined)
    
master_df = pd.concat(master_df)
master_df.to_excel("google_results.xls")

