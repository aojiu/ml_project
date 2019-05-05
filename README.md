# ml_project
machine learning 2019 spring project

## Object
using news, exchange rate, ralevence stock to pridict the Apple's stock price

## General method
Collecting the news from 2011 to 2017. Then using sentiment analysis to evaluate the articals. Next processing the features from news analysis into the same days. Meanwhile, we have collected data contains exchange rate, ralevence stocks price, and Apple stock price. Now we have to merge news, exchange rate, ralevence stocks price with same day as features. Apple stock price is target. For each day that stock price and if it's prior day also has the stock price, then this day target would have features value of prior day. However, if it's prior day has no stock price, the feature would be average of all priors no stock days' features.  <br>

## Dataset:
### News:
url : https://www.kaggle.com/snapcrack/all-the-news#articles3.csv <br>
In this dataset, there could be many articals in the same days. <br>
#### Features:
title, publication, author, date, year, month, url, content <br>
We only use the content to do the sentiment analysis, and keep the date only. 
#### Sentiment Analysis:
We using Empath sentiment anaylsis.<br>
github: https://github.com/Ejhfast/empath-client<br>
 
