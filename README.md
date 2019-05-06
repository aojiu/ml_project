# Machine Learning Project
machine learning 2019 spring project

## Object
using news, exchange rate, ralevence stock to pridict the Apple's stock price change

## General method
Collecting the news from 2011 to 2017. Then using sentiment analysis to evaluate the articals. Next processing the features from news analysis into the same days. Meanwhile, we have collected data contains exchange rate, ralevence stocks price, and Apple stock price. Now we have to merge news, exchange rate, ralevence stocks price with same day as features. Apple stock price is target. For each day that stock price and if it's prior day also has the stock price, then this day target would have features value of prior day. However, if it's prior day has no stock price, the feature would be average of all priors no stock days' features.  <br>

## Dataset Description and Pre-processing:

### News Dataset
#### Original News Dataset:
url : https://www.kaggle.com/snapcrack/all-the-news#articles3.csv <br>
In this dataset, there could be many articals in the same days. <br>

#### Features:
title, publication, author, date, year, month, url, content <br>
We only use the content to do the sentiment analysis, and keep the date only. 

#### Pre-processing News Dataset:
##### Sentiment Analysis:
We using Empath sentiment anaylsis.<br>
github: https://github.com/Ejhfast/empath-client<br>
The sentiment analysis will analysis a string type of artical, and return a value of how ralevence to the target word.
###### example:
	lexicon.analyze("he hit the other person", normalize=True)
	# => {'help': 0.0, 'office': 0.0, 'violence': 0.2, 'dance': 0.0, 'money': 0.0, 'wedding': 0.0, ... )
we use part of this return values as features for news. <br>
##### Features for creating new news dataset
There three groups features. <br>
First group contains features that show how much a artical is relevant to the Apple Company<br>
######
	"social_media","computer","business","programming","hearing","urban",
	"shopping","science","work","valuable","fashion","technology","competing","economics","office"
	
Second group contains features that show the negative emotions<br>
#####
	'hate','aggression','horror','suffering','ridicule','irritability',
	'deception','disappointment','negative_emotion','nervousness'
Third group contains features that show the positive emotions<br>
#####
	'cheerfulness','optimism','celebration','trust','positive_emotion'
Next we merge these values for each article to each day since there many articles for one day.<br>
##### Separate into two datasets
Now we have a dataset with three group features, but we want process more. Thus we create three new features, which are<br>
######
	relevance, positive_relevant, negative_relevant
relevance = avg(relevant group features value)<br>
positive_relevant = relevance * avg(positive group features value)<br>
negative_relevant = relevance * avg(negative group features value)<br>
The reason for creating three features is that we want explore the relevant artical with such emotions in general would have any relation with the stock price. <br>
As result, we have two dataset. One contains all three group features, and Other only contains three features.

### Exchange Rate Dataset:
https://github.com/datasets/exchange-rates/tree/master/data
#### Features:
	Euro exchange rate, Hong Kong exchange rate, Canada exchange rate, China exchange rate
As we consider that Apple is global company, so when exchange rate changes, the sales rate toward to other countries would changes as well, which may influence the stock prices. 

### Relevant Stock Price:
#### Features:

### APPLE Stock price as Y:
https://finance.yahoo.com/quote/AAPL/history/ <br>
plz write here how to process the y, detailed.<br>

### Creat Two diffent typs y:
for different model, we need different target value. Such as that we need continuous number for linear regression, but we need binary number for SVM and Neural Network models.<br>
As result, the target for continuous number is close price of each day for apple and the difference between close price and open price of day. The target for discrete number is when close price - open price less than 0, or more than 0.

## Linear Regression:
we used Polynomial Linear Regression with Ridge <br>
